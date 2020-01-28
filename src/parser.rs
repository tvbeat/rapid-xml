//! Contains low-level XML `Parser`.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::borrow::Cow;
use std::convert::TryInto;
use std::fmt::Display;
use std::io::Read;

use inlinable_string::InlinableString;
use slice_deque::SliceDeque;

use multiversion::multiversion;

const READ_SIZE: usize = 4 * 4096; // TODO: Fine-tune.
const BLOCK_SIZE: usize = 64; // Size of u64, 64 characters, 4 sse2 128i loads, 2 avx 256i loads.


/// Fills the `positions` vector with indexes of bytes containing the control characters.
///
/// Searches the `input` but only from the `start` index forwards. If there was anything in the
/// `positions` vector, the new values are appended behind.
#[inline(always)]
fn classify_fallback(input: &[u8], start: usize, positions: &mut Vec<usize>) {
    for (i, c) in input[start..].iter().enumerate() {
        match c {
            b'\t' | b'\n' | b'\r' | b' ' | b'\"' | b'\'' | b'<' | b'=' | b'>' | b'&' =>
                positions.push(i + start),
            _ => {}
        }
    }
}

/// Same as `classify_fallback`, but implemented using SIMD intrinsics.
///
/// # Safety
///
/// Can be only called if SSSE3 is available.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn classify_ssse3(input: &[u8], start: usize, positions: &mut Vec<usize>) {
    // SIMD classification using two `pshufb` instructions (`_mm_shuffle_epi8` intrinsic) that match
    // the high and low nibble of the byte. Combining that together we get a non-zero value for every
    // of the matched characters.

    //     hi / lo
    //         +--------------------------------
    //         | 0 1 2 3 4 5 6 7 8 9 a b c d e f
    //       --+--------------------------------
    //       0 | . . . . . . . . . A A . . A . .
    //       1 | . . . . . . . . . . . . . . . .
    //       2 | B . " . . . & ' . . . . . . . /
    //       3 | . . . . . . . . . . . . < = > .
    //       4 | . . . . . . . . . . . . . . . .
    //       5 | . . . . . . . . . . . . . . . .
    //       6 | . . . . . . . . . . . . . . . .
    //       7 | . . . . . . . . . . . . . . . .
    //       8 | . . . . . . . . . . . . . . . .
    //       9 | . . . . . . . . . . . . . . . .
    //       a | . . . . . . . . . . . . . . . .
    //       b | . . . . . . . . . . . . . . . .
    //       c | . . . . . . . . . . . . . . . .
    //       d | . . . . . . . . . . . . . . . .
    //       e | . . . . . . . . . . . . . . . .
    //       f | . . . . . . . . . . . . . . . .


    const NOTHING: i8      = 0;
    const SPACE_A: i8      = (1 << 0);
    const SPACE_B: i8      = (1 << 1);
    const DOUBLE_QUOTE: i8 = (1 << 2);
    const SINGLE_QUOTE: i8 = (1 << 3);
    const AMPERSAND: i8    = (1 << 4);
    const LQ: i8           = (1 << 5);
    const EQ: i8           = (1 << 6);
    const GT: i8           = (1 << 7);

    let lo_nibbles_lookup = _mm_setr_epi8(
        /* 0 */ SPACE_B,
        /* 1 */ NOTHING,
        /* 2 */ DOUBLE_QUOTE,
        /* 3 */ NOTHING,
        /* 4 */ NOTHING,
        /* 5 */ NOTHING,
        /* 6 */ AMPERSAND,
        /* 7 */ SINGLE_QUOTE,
        /* 8 */ NOTHING,
        /* 9 */ SPACE_A,
        /* a */ SPACE_A,
        /* b */ NOTHING,
        /* c */ LQ,
        /* d */ SPACE_A | EQ,
        /* e */ GT,
        /* f */ NOTHING,
    );

    let hi_nibbles_lookup = _mm_setr_epi8(
        /* 0 */ SPACE_A,
        /* 1 */ NOTHING,
        /* 2 */ SPACE_B | DOUBLE_QUOTE | SINGLE_QUOTE | AMPERSAND,
        /* 3 */ LQ | EQ | GT,
        /* 4 */ NOTHING,
        /* 5 */ NOTHING,
        /* 6 */ NOTHING,
        /* 7 */ NOTHING,
        /* 8 */ NOTHING,
        /* 9 */ NOTHING,
        /* a */ NOTHING,
        /* b */ NOTHING,
        /* c */ NOTHING,
        /* d */ NOTHING,
        /* e */ NOTHING,
        /* f */ NOTHING,
    );

    let mut ptr = input[start..].as_ptr();
    let end = input.as_ptr().offset(input.len() as isize);
    debug_assert!(ptr as usize % BLOCK_SIZE == 0);
    debug_assert!(end as usize % BLOCK_SIZE == 0);

    while ptr < end {
        let input_a = _mm_load_si128(ptr as *const __m128i);
        let lo_nibbles = _mm_and_si128(input_a, _mm_set1_epi8(0x0f));
        let hi_nibbles = _mm_and_si128(_mm_srli_epi16(input_a, 4), _mm_set1_epi8(0x0f));
        let lo_translated = _mm_shuffle_epi8(lo_nibbles_lookup, lo_nibbles);
        let hi_translated = _mm_shuffle_epi8(hi_nibbles_lookup, hi_nibbles);
        let intersection = _mm_and_si128(lo_translated, hi_translated);
        let eq = _mm_cmpeq_epi8(intersection, _mm_set1_epi8(0)); // TODO: Better way to do this line?
        let mask_a = (_mm_movemask_epi8(eq) as u16) as u64;

        let input_b = _mm_load_si128(ptr.offset(16) as *const __m128i);
        let lo_nibbles = _mm_and_si128(input_b, _mm_set1_epi8(0x0f));
        let hi_nibbles = _mm_and_si128(_mm_srli_epi16(input_b, 4), _mm_set1_epi8(0x0f));
        let lo_translated = _mm_shuffle_epi8(lo_nibbles_lookup, lo_nibbles);
        let hi_translated = _mm_shuffle_epi8(hi_nibbles_lookup, hi_nibbles);
        let intersection = _mm_and_si128(lo_translated, hi_translated);
        let eq = _mm_cmpeq_epi8(intersection, _mm_set1_epi8(0)); // TODO: Better way to do this line?
        let mask_b = (_mm_movemask_epi8(eq) as u16) as u64;

        let input_c = _mm_load_si128(ptr.offset(32) as *const __m128i);
        let lo_nibbles = _mm_and_si128(input_c, _mm_set1_epi8(0x0f));
        let hi_nibbles = _mm_and_si128(_mm_srli_epi16(input_c, 4), _mm_set1_epi8(0x0f));
        let lo_translated = _mm_shuffle_epi8(lo_nibbles_lookup, lo_nibbles);
        let hi_translated = _mm_shuffle_epi8(hi_nibbles_lookup, hi_nibbles);
        let intersection = _mm_and_si128(lo_translated, hi_translated);
        let eq = _mm_cmpeq_epi8(intersection, _mm_set1_epi8(0)); // TODO: Better way to do this line?
        let mask_c = (_mm_movemask_epi8(eq) as u16) as u64;

        let input_d = _mm_load_si128(ptr.offset(48) as *const __m128i);
        let lo_nibbles = _mm_and_si128(input_d, _mm_set1_epi8(0x0f));
        let hi_nibbles = _mm_and_si128(_mm_srli_epi16(input_d, 4), _mm_set1_epi8(0x0f));
        let lo_translated = _mm_shuffle_epi8(lo_nibbles_lookup, lo_nibbles);
        let hi_translated = _mm_shuffle_epi8(hi_nibbles_lookup, hi_nibbles);
        let intersection = _mm_and_si128(lo_translated, hi_translated);
        let eq = _mm_cmpeq_epi8(intersection, _mm_set1_epi8(0)); // TODO: Better way to do this line?
        let mask_d = (_mm_movemask_epi8(eq) as u16) as u64;

        let mut mask = !(mask_a | (mask_b << 16) | (mask_c << 32) | (mask_d << 48));

        let count = mask.count_ones() as usize;
        positions.reserve(BLOCK_SIZE); // Max amount of control characters we may find.
        let mut out = positions.as_mut_ptr().offset(positions.len() as isize);

        // To reduce the amount of branches, we retrieve bit indexes from `mask` in batches of 4,
        // later we will `set_len` to the appropriate amount. If the amount was not multiple of 4,
        // the extra values are incorrect, but later ignored.
        while mask != 0 {
            // TODO: What amount of repetitions is best?

            *out = mask.trailing_zeros() as usize + (ptr as usize - input.as_ptr() as usize);
            out = out.offset(1);
            mask = mask & mask.overflowing_sub(1).0;

            *out = mask.trailing_zeros() as usize + (ptr as usize - input.as_ptr() as usize);
            out = out.offset(1);
            mask = mask & mask.overflowing_sub(1).0;

            *out = mask.trailing_zeros() as usize + (ptr as usize - input.as_ptr() as usize);
            out = out.offset(1);
            mask = mask & mask.overflowing_sub(1).0;

            *out = mask.trailing_zeros() as usize + (ptr as usize - input.as_ptr() as usize);
            out = out.offset(1);
            mask = mask & mask.overflowing_sub(1).0;
        }

        positions.set_len(positions.len() + count);

        ptr = ptr.offset(BLOCK_SIZE as isize);
    }
}

/// Same as `classify_ssse3`, but using AVX2 instructions.
///
/// # Safety
///
/// Can be only called if AVX2 is available.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
#[allow(unused)]
unsafe fn classify_avx2(input: &[u8], start: usize, positions: &mut Vec<usize>) {
    // Same algorithm as classify_ssse3, but working on half the amount of twice as long vectors.

    const NOTHING: i8      = 0;
    const SPACE_A: i8      = (1 << 0);
    const SPACE_B: i8      = (1 << 1);
    const DOUBLE_QUOTE: i8 = (1 << 2);
    const SINGLE_QUOTE: i8 = (1 << 3);
    const AMPERSAND: i8    = (1 << 4);
    const LQ: i8           = (1 << 5);
    const EQ: i8           = (1 << 6);
    const GT: i8           = (1 << 7);

    let lo_nibbles_lookup = _mm256_set_epi8(
        /* 0 */ SPACE_B,
        /* 1 */ NOTHING,
        /* 2 */ DOUBLE_QUOTE,
        /* 3 */ NOTHING,
        /* 4 */ NOTHING,
        /* 5 */ NOTHING,
        /* 6 */ AMPERSAND,
        /* 7 */ SINGLE_QUOTE,
        /* 8 */ NOTHING,
        /* 9 */ SPACE_A,
        /* a */ SPACE_A,
        /* b */ NOTHING,
        /* c */ LQ,
        /* d */ SPACE_A | EQ,
        /* e */ GT,
        /* f */ NOTHING,

        /* 0 */ SPACE_B,
        /* 1 */ NOTHING,
        /* 2 */ DOUBLE_QUOTE,
        /* 3 */ NOTHING,
        /* 4 */ NOTHING,
        /* 5 */ NOTHING,
        /* 6 */ AMPERSAND,
        /* 7 */ SINGLE_QUOTE,
        /* 8 */ NOTHING,
        /* 9 */ SPACE_A,
        /* a */ SPACE_A,
        /* b */ NOTHING,
        /* c */ LQ,
        /* d */ SPACE_A | EQ,
        /* e */ GT,
        /* f */ NOTHING,
    );

    let hi_nibbles_lookup = _mm256_set_epi8(
        /* 0 */ SPACE_A,
        /* 1 */ NOTHING,
        /* 2 */ SPACE_B | DOUBLE_QUOTE | SINGLE_QUOTE | AMPERSAND,
        /* 3 */ LQ | EQ | GT,
        /* 4 */ NOTHING,
        /* 5 */ NOTHING,
        /* 6 */ NOTHING,
        /* 7 */ NOTHING,
        /* 8 */ NOTHING,
        /* 9 */ NOTHING,
        /* a */ NOTHING,
        /* b */ NOTHING,
        /* c */ NOTHING,
        /* d */ NOTHING,
        /* e */ NOTHING,
        /* f */ NOTHING,

        /* 0 */ SPACE_A,
        /* 1 */ NOTHING,
        /* 2 */ SPACE_B | DOUBLE_QUOTE | SINGLE_QUOTE | AMPERSAND,
        /* 3 */ LQ | EQ | GT,
        /* 4 */ NOTHING,
        /* 5 */ NOTHING,
        /* 6 */ NOTHING,
        /* 7 */ NOTHING,
        /* 8 */ NOTHING,
        /* 9 */ NOTHING,
        /* a */ NOTHING,
        /* b */ NOTHING,
        /* c */ NOTHING,
        /* d */ NOTHING,
        /* e */ NOTHING,
        /* f */ NOTHING,
    );

    let mut ptr = input[start..].as_ptr();
    let end = input.as_ptr().offset(input.len() as isize);
    debug_assert!(ptr as usize % BLOCK_SIZE == 0);
    debug_assert!(end as usize % BLOCK_SIZE == 0);

    while ptr < end {
        let input_a = _mm256_load_si256(ptr as *const __m256i);
        let lo_nibbles = _mm256_and_si256(input_a, _mm256_set1_epi8(0x0f));
        let hi_nibbles = _mm256_and_si256(_mm256_srli_epi16(input_a, 4), _mm256_set1_epi8(0x0f));
        let lo_translated = _mm256_shuffle_epi8(lo_nibbles_lookup, lo_nibbles);
        let hi_translated = _mm256_shuffle_epi8(hi_nibbles_lookup, hi_nibbles);
        let intersection = _mm256_and_si256(lo_translated, hi_translated);
        let eq = _mm256_cmpeq_epi8(intersection, _mm256_set1_epi8(0)); // TODO: Better way to do this line?
        let mask_a = (_mm256_movemask_epi8(eq) as u32) as u64;

        let input_b = _mm256_load_si256(ptr.offset(32) as *const __m256i);
        let lo_nibbles = _mm256_and_si256(input_b, _mm256_set1_epi8(0x0f));
        let hi_nibbles = _mm256_and_si256(_mm256_srli_epi16(input_b, 4), _mm256_set1_epi8(0x0f));
        let lo_translated = _mm256_shuffle_epi8(lo_nibbles_lookup, lo_nibbles);
        let hi_translated = _mm256_shuffle_epi8(hi_nibbles_lookup, hi_nibbles);
        let intersection = _mm256_and_si256(lo_translated, hi_translated);
        let eq = _mm256_cmpeq_epi8(intersection, _mm256_set1_epi8(0)); // TODO: Better way to do this line?
        let mask_b = (_mm256_movemask_epi8(eq) as u32) as u64;

        let mut mask = !(mask_a | (mask_b << 32));

        let count = mask.count_ones() as usize;
        positions.reserve(BLOCK_SIZE); // Max amount of control characters we may find.
        let mut out = positions.as_mut_ptr().offset(positions.len() as isize);

        while mask != 0 {
            // TODO: What amount of repetitions is best?

            *out = mask.trailing_zeros() as usize + (ptr as usize - input.as_ptr() as usize);
            out = out.offset(1);
            mask = mask & mask.overflowing_sub(1).0;

            *out = mask.trailing_zeros() as usize + (ptr as usize - input.as_ptr() as usize);
            out = out.offset(1);
            mask = mask & mask.overflowing_sub(1).0;

            *out = mask.trailing_zeros() as usize + (ptr as usize - input.as_ptr() as usize);
            out = out.offset(1);
            mask = mask & mask.overflowing_sub(1).0;

            *out = mask.trailing_zeros() as usize + (ptr as usize - input.as_ptr() as usize);
            out = out.offset(1);
            mask = mask & mask.overflowing_sub(1).0;
        }

        positions.set_len(positions.len() + count);

        ptr = ptr.offset(BLOCK_SIZE as isize);
    }
}

multiversion! {
    fn classify(input: &[u8], start: usize, positions: &mut Vec<usize>)

    // Order is from the best to worst.
//    "[x86|x86_64]+avx2" => classify_avx2, // Somehow this turns out to be even slower than classify_fallback.
    "[x86|x86_64]+ssse3" => classify_ssse3,
    default => classify_fallback,
}

/// The kind of XML malformation that was encountered
#[derive(Debug)]
pub enum MalformedXMLKind {
    /// Tag name contained invalid characters
    BadTagName,

    /// Attribute name contained invalid characters
    BadAttributeName,

    /// Attribute value contained invalid characters or was not properly quoted
    BadAttributeValue,

    /// Closing tag contained attributes or other extra character
    ExtrasInClosingTag,

    /// The character '<' outside of beginning of a tag was not escaped (e.g. in string)
    UnescapedLessThan,

    /// EOF before the XML structure was properly terminated
    UnexpectedEof,
}

/// Error while parsing
#[derive(Debug)]
pub enum ParseError {
    /// IO error reading from the underlying Read
    IO(std::io::Error),

    /// Error converting string to Utf8
    Utf8(std::str::Utf8Error),

    /// XML was not well formed
    MalformedXML {
        /// Byte index of the problem
        byte: Option<usize>,

        /// Offending character, if any
        character: Option<u8>,

        /// Kind of malformation
        kind: MalformedXMLKind,
    },
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        match self {
            ParseError::IO(err) => Display::fmt(err, f),
            ParseError::Utf8(err) => Display::fmt(err, f),
            ParseError::MalformedXML { byte, character, kind } => {
                let msg = match kind {
                    MalformedXMLKind::BadTagName => "bad tag name",
                    MalformedXMLKind::BadAttributeName => "bad attribute name",
                    MalformedXMLKind::BadAttributeValue => "bad value name",
                    MalformedXMLKind::ExtrasInClosingTag => "extra things in closing tag",
                    MalformedXMLKind::UnescapedLessThan => "unescaped '<' character",
                    MalformedXMLKind::UnexpectedEof => "unexpected EOF",
                };

                write!(f, "Malformed XML: {}", msg)?;
                if let Some(byte) = byte {
                    write!(f, ", byte {}", byte)?;
                }
                if let Some(character) = character {
                    write!(f, ", character {:?}", character)?;
                }

                Ok(())
            }
        }
    }
}

impl std::error::Error for ParseError {}

impl From<std::io::Error> for ParseError {
    fn from(err: std::io::Error) -> Self {
        ParseError::IO(err)
    }
}

impl From<std::str::Utf8Error> for ParseError {
    fn from(err: std::str::Utf8Error) -> Self {
        ParseError::Utf8(err)
    }
}

/// A string borrowed from the XML stream
///
/// Verification of Utf-8 correctness, decoding of character entities, EOL normalizing and other
/// possible transformations are done lazily.
#[derive(Clone, Copy, Debug, Eq)]
pub struct DeferredString<'a> {
    bytes: &'a [u8],
    needs_decoding: bool,
    needs_eol_normalizing: bool,
}

impl<'a> DeferredString<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self {
            bytes,
            needs_decoding: false,
            needs_eol_normalizing: false,
        }
    }

    fn with_options(bytes: &'a [u8], needs_decoding: bool, needs_eol_normalizing: bool) -> Self {
        Self {
            bytes,
            needs_decoding,
            needs_eol_normalizing,
        }
    }

    /// Builds `DeferredString` from Rust string
    ///
    /// This is only needed if you need to synthetize `DeferredString`, for example for test.
    pub fn from_str(string: &'a str) -> Self {
        Self {
            bytes: string.as_bytes(),
            needs_decoding: false,
            needs_eol_normalizing: false,
        }
    }

    /// Convert to string in Rust representation
    ///
    /// This may return error in case of invalid Utf-8 or some other problems.
    pub fn to_str(&self) -> Result<Cow<str>, ParseError> {
        // TODO: Handle decoding and eol normalizing!
        Ok(std::str::from_utf8(self.bytes).map(Cow::Borrowed)?)
    }
}

impl<'a> TryInto<InlinableString> for DeferredString<'a> {
    type Error = ParseError;

    fn try_into(self) -> Result<InlinableString, Self::Error> {
        match self.to_str()? {
            Cow::Borrowed(str) => Ok(InlinableString::from(str)),
            Cow::Owned(string) => Ok(InlinableString::from(string)),
        }
    }
}

impl<'a> PartialEq for DeferredString<'a> {
    fn eq(&self, other: &Self) -> bool {
        if !self.needs_decoding &&
            !self.needs_eol_normalizing &&
            !other.needs_decoding &&
            !other.needs_eol_normalizing
        {
            self.bytes == other.bytes
        } else {
            match (self.to_str(), other.to_str()) {
                (Ok(left), Ok(right)) => left == right,
                _ => false,
            }
        }
    }
}

/// Low-level XML event, marks a significant part of XML document
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Event<'a> {
    /// Opening of a tag, the `<tag_name` portion of it. The string contains the `tag_name` part.
    StartTag(DeferredString<'a>),

    /// This is emitted when the opening tag ends, but is not immediately terminated.
    /// It is emitted at the `>` character of `<tag_name bla='ble'>`.
    /// If the tag is immediatelly terminated (`<tag_name bla='ble'/>`), the `EndTagImmediate` is
    /// emitted instead!
    StartTagDone,

    /// Self-standing closing tag. E.g. `</tag_name>`. The string contains the `tag_name` part.
    EndTag(DeferredString<'a>),

    /// The immediate closing of a tag at the end of a start tag.
    /// It is the `/>` at the end of `<tag_name bla='ble'/>`.
    EndTagImmediate,

    /// Any non-whitespace text in between tags, stripped.
    Text(DeferredString<'a>),

    /// The attribute name, from ` bla = 'ble' `, the string contains `bla`.
    /// In well formed XML it is always followed by `AttributeValue`.
    AttributeName(DeferredString<'a>),

    /// The attribute value, from ` bla = 'ble' `, the string contains `ble`.
    /// In well formed XML it always follows after `AttributeValue`.
    AttributeValue(DeferredString<'a>),

    /// End of file. Emitted when the underlying `Read` reaches end and will be emitted indefinitely
    /// after that.
    Eof,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum State {
    InText,
    AtTagStart,
    AtTagStartDone,
    InTag,
    InTagAfterAttributeName,
    AtImmediateTagEnd,
}

/// A low level XML parser that emits `Event`s as it reads the incoming XML.
pub struct Parser<R: Read> {
    // The source of input data
    reader: R,

    // Ring buffer that is always continuous in memory thanks to clever memory mapping
    buffer: SliceDeque<u8>,

    // The index of the byte at the beginning of `buffer`. In other words, the amount of bytes that
    // we thrown away because they were no longer needed in `buffer`.
    buffer_starts_at: usize,

    // Indexes into the `buffer.as_slice()` of positions of control characters
    control_chars: Vec<usize>,

    // How many of `control_chars` were already read
    control_chars_read: usize,

    // Index into the `buffer.as_slice()` marking the furthest point up to where we parsed to.
    // The resulting address should be multiple of BLOCK_SIZE.
    parsed_up_to: usize,

    // Was EOF already reached in the input reader?
    reached_eof: bool,

    // State of the parser
    state: State,

    // Index of the last control character
    last_index: usize,
}

impl<R: Read> Parser<R> {
    /// Create new `Parser` from given reader.
    pub fn new(reader: R) -> Self {
        Self {
            reader,
            buffer: SliceDeque::with_capacity(READ_SIZE * 2),
            buffer_starts_at: 0,
            control_chars: Vec::new(),
            control_chars_read: 0,
            parsed_up_to: 0,
            reached_eof: false,
            state: State::InText,
            last_index: 0,
        }
    }

    /// Read additional (roughly) `READ_SIZE` amount of data and parse it to fill the `self.object_indexes`.
    ///
    /// Caller should first remove all used control characters from `control_chars`
    fn classify_more(&mut self) -> Result<(), std::io::Error> {
        // First we throw away the part of buffer that we no longer need.
        let to_throw_away = ((self.last_index - self.buffer_starts_at) / BLOCK_SIZE) * BLOCK_SIZE; // In theory we could throw away all `last_index` bytes, but if the SliceDeque decides to reallocate, it would shift the data such that the `head` is at beginning of the page, which would break our alignment, because `last_index` and `parsed_up_to` are not generally multiple of BLOCK_SIZE away.
        unsafe { // Safety: We know there is at least this much in the buffer
            self.buffer.move_head_unchecked(to_throw_away as isize);
        }
        self.parsed_up_to -= to_throw_away;
        for control_char in &mut self.control_chars { // Usually there shouldn't be anything in there, unless we are called from `get`
            *control_char -= to_throw_away;
        }
        self.buffer_starts_at += to_throw_away;

        // Then we read into the buffer
        self.buffer.reserve(READ_SIZE); // This won't do anything if we already grew enough, unless we have some long unfinished json object still in the buffer.

        unsafe {
            let mut total_bytes_got = 0;
            loop {
                let uninit_buffer = self.buffer.tail_head_slice();
                // TODO, XXX: In general putting uninit buffer to `Read::read` is not allowed! The right way
                //            to do this is still being discussed all around Rust forums. But this normally
                //            works because no sane `Read`er reads from the target buffer, it just writes into
                //            it. It will typically end up given to libc `read` function that does not care
                //            about unitialized bytes.
                let bytes_got = self.reader.read(uninit_buffer)?;

                if bytes_got == 0 {
                    self.reached_eof = true;

                    // We reached the end of file, so we are not going to get any more data from this reader.
                    // But to finish parsing what we have, we need our input data to be padded to BLOCK_SIZE,
                    // so we append spaces behind what we got until BLOCK_SIZE.
                    let padding = (BLOCK_SIZE - (self.buffer.as_ptr() as usize + self.buffer.len()) % BLOCK_SIZE) % BLOCK_SIZE;

                    for _ in 0..padding {
                        self.buffer.push_back(b'\0');
                    }

                    break;
                }

                self.buffer.move_tail_unchecked(bytes_got as isize);

                // In order to use the underlying `Read` efficiently, we do not require exact amount,
                // but we keep reading until we got at least half of READ_SIZE (or EOF), so that we
                // can do efficient parsing in big batches.
                total_bytes_got += bytes_got;
                if total_bytes_got > READ_SIZE / 2 {
                    break;
                }
            }
        }

        // Then we parse the new data, or at least the BLOCK_SIZE-aligned part of it. We may leave
        // few bytes (0..BLOCK_SIZE) not parsed, it will be parsed next time when we read more data
        // behind it.
        let slice = self.buffer.as_slice();
        let previously_parsed_up_to = self.parsed_up_to;
        self.parsed_up_to = slice.len() / BLOCK_SIZE * BLOCK_SIZE;
        let aligned_slice = &slice[0..self.parsed_up_to];

        classify(aligned_slice, previously_parsed_up_to, &mut self.control_chars);

        // Now we **may** have some additional control characters, lets try again
        Ok(())
    }

    /// Get next control character and its index
    ///
    /// Returns nul character if EOF was reached. This is ok because well formed XML can not contain
    /// nul character.
    fn next_control_char(&mut self) -> Result<(u8, usize), std::io::Error> {
        while self.control_chars_read == self.control_chars.len() && !self.reached_eof {
            self.control_chars.clear();
            self.control_chars_read = 0;
            self.classify_more()?;
        }

        if self.control_chars_read < self.control_chars.len() {
            let index = self.control_chars[self.control_chars_read];
            self.control_chars_read += 1;

            return Ok((self.buffer[index], index + self.buffer_starts_at));
        }

        Ok((b'\0', self.buffer.len() + self.buffer_starts_at))
    }

    /// Get byte at given index.
    fn get(&mut self, index: usize) -> Result<u8, std::io::Error> {
        loop {
            // Note that this must be here inside loop, because if we call classify_more, we need to
            // do it over again.
            let index = match index.checked_sub(self.buffer_starts_at) {
                Some(index) => index,
                None => unreachable!("Asking for data that are no longer in buffer!"),
            };

            match self.buffer.get(index) {
                Some(c) => return Ok(*c),
                None => {
                    if self.reached_eof {
                        return Ok(b'\0');
                    }
                    // Note that following will have to shift the tail and VecDeque would be better
                    // at this. But this happens rarely (that get() attempts to read after the last
                    // character of the buffer). Using Vec instead of VecDeque allows us to
                    // efficiently add to it in classify_ssse3 (and possibly others), which is more
                    // important.
                    self.control_chars.drain(0..self.control_chars_read);
                    self.control_chars_read = 0;
                    self.classify_more()?;
                }
            }
        }
    }

    fn get_slice(&self, from: usize, to: usize) -> &[u8] {
        let (from, to) = match (from.checked_sub(self.buffer_starts_at),
                                to.checked_sub(self.buffer_starts_at)) {
            (Some(from), Some(to)) => (from, to),
            _ => unreachable!("Asking for data that are no longer in buffer!"),
        };

        &self.buffer[from..to]
    }

    /// Retrieve next `Event`
    pub fn next(&mut self) -> Result<Event, ParseError> {
        let event = self.next_();

//        match &event {
//            Ok(Event::StartTag(text)) => eprintln!("StartTag({:?})", text.to_str().unwrap()),
//            Ok(Event::StartTagDone) => eprintln!("StartTagDone"),
//            Ok(Event::EndTag(text)) => eprintln!("EndTag({:?})", text.to_str().unwrap()),
//            Ok(Event::EndTagImmediate) => eprintln!("EndTagImmediate"),
//            Ok(Event::AttributeName(text)) => eprintln!("AttributeName({:?})", text.to_str().unwrap()),
//            Ok(Event::AttributeValue(text)) => eprintln!("AttributeValue({:?})", text.to_str().unwrap()),
//            Ok(Event::Text(text)) => eprintln!("Text({:?})", text.to_str().unwrap()),
//            Ok(Event::Eof) => eprintln!("Eof"),
//            Err(err) => eprintln!("Err({:?})", err),
//        }

        event
    }

    fn next_(&mut self) -> Result<Event, ParseError> {
        // Nested state machines that turn sequence of control characters into events.
        loop {
            match self.state {
                // If we are inside text (e.g. outside any tag < >)...
                State::InText => {
                    loop {
                        match self.next_control_char()? {
                            // If next is start of a tag, we may have some string that should be
                            // emitted and next time we'll read the tag. If not, we just straight
                            // into reading that tag.
                            (b'<', i) => {
                                let from = self.last_index + 1;
                                self.last_index = i;
                                self.state = State::AtTagStart;

                                if from < i {
                                    return Ok(Event::Text(DeferredString::new(self.get_slice(from, i))));
                                } else {
                                    break;
                                }
                            }

                            // EOF? That is ok in this state!
                            (b'\0', _) => return Ok(Event::Eof),

                            // If we see continous whitespace characters at the beginning of text,
                            // we just shift the beginning of the text to effectively strip it.
                            (b'\t', i) | (b'\n', i) | (b'\r', i) | (b' ', i)
                                if (self.last_index == 0 && i == 0) || self.last_index + 1 == i =>
                            {
                                self.last_index = i;
                            }

                            // If we encounter any other control characters, we must go deeper...
                            (c, i) => {
                                let mut needs_eol_normalizing = false;
                                let mut needs_decoding = false;
                                let mut last_whitespace = self.last_index;
                                let mut last_non_whitespace = self.last_index - 1;
                                if let b'\t' | b'\n' | b'\r' | b' ' = c {
                                    last_whitespace = i;
                                    last_non_whitespace = i - 1;
                                }
                                loop {
                                    match self.next_control_char()? {
                                        // The presence of carriage return character means that we
                                        // will need EOL normalizing. Also we must track the continous
                                        // whitespace at the end of the text to be able to strip it.
                                        (b'\r', i) => {
                                            needs_eol_normalizing = true;

                                            if last_whitespace + 1 != i {
                                                last_non_whitespace = i - 1;
                                            }
                                            last_whitespace = i;
                                        }

                                        // Track the continous whitespace at the end of the text for
                                        // stripping.
                                        (b'\t', i) | (b'\n', i) | (b' ', i) => {
                                            if last_whitespace + 1 != i {
                                                last_non_whitespace = i - 1;
                                            }
                                            last_whitespace = i;
                                        }

                                        // The presence of amphersand means we will need decoding.
                                        // Also it is a non-whitespace character.
                                        (b'&', i) => {
                                            needs_decoding = true;
                                            last_non_whitespace = i;
                                        }

                                        // The beginning of a tag ends our text. If there was any
                                        // text apart from whitespace, we must emit it. Otherwise we
                                        // just jump to tag processing.
                                        (b'<', i) => {
                                            if last_whitespace + 1 != i {
                                                last_non_whitespace = i - 1;
                                            }
                                            let from = self.last_index + 1;
                                            self.last_index = i;
                                            self.state = State::AtTagStart;
                                            if from < last_non_whitespace {
                                                return Ok(Event::Text(DeferredString::with_options(
                                                    self.get_slice(from, last_non_whitespace + 1),
                                                    needs_decoding,
                                                    needs_eol_normalizing,
                                                )));
                                            } else {
                                                break;
                                            }
                                        }

                                        // EOF is ok in text
                                        (b'\0', _) => return Ok(Event::Eof),

                                        // Any other control character - remember position of a
                                        // non-whitespace character
                                        _ => {
                                            last_non_whitespace = i;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // If we are at the start of a tag
                State::AtTagStart => {
                    match self.get(self.last_index + 1)? {
                        // Is it a processing instruction? Then we skip it...
                        b'?' => {
                            loop {
                                let (c, i) = self.next_control_char()?;
                                if c == b'>' && self.get(i - 1)? == b'?' {
                                    self.last_index = i;
                                    self.state = State::InText;
                                    break;
                                }
                                if c == b'\0' {
                                    return Err(ParseError::MalformedXML {
                                        byte: Some(i),
                                        character: None,
                                        kind: MalformedXMLKind::UnexpectedEof,
                                    });
                                }
                            }
                        }

                        // Is it a comment, CDATA or something else? Oops, we can't handle that yet.
                        b'!' => unimplemented!("TODO: Handle <! tags"),

                        // Is it closing tag? Find the end of it and report it.
                        b'/' => {
                            let from = self.last_index + 2;
                            let mut to = None;
                            loop {
                                match self.next_control_char()? {
                                    (b'\t', i) | (b'\n', i) | (b'\r', i) | (b' ', i) => {
                                        to.get_or_insert(i);
                                    }

                                    (b'>', i) => {
                                        let to = to.get_or_insert(i);
                                        self.last_index = i;
                                        self.state = State::InText;
                                        return Ok(Event::EndTag(DeferredString::new(self.get_slice(from, *to))));
                                    }

                                    (c, i) => return Err(ParseError::MalformedXML {
                                        byte: Some(i),
                                        character: Some(c),
                                        kind: MalformedXMLKind::ExtrasInClosingTag,
                                    }),
                                }
                            }
                        }

                        // So it is normal opening tag...
                        _ => {
                            match self.next_control_char()? {
                                // Whitespace inside means we found the end of the tag's name, report it.
                                (b'\t', i) | (b'\n', i) | (b'\r', i) | (b' ', i) => {
                                    let from = self.last_index + 1;
                                    self.last_index = i;
                                    self.state = State::InTag;
                                    return Ok(Event::StartTag(DeferredString::new(self.get_slice(from, i))));
                                }

                                // Immediate '>', the tag is over right after the tag's name. Report that
                                // and remember to report either TagStartDone or TagEndImmediate next time.
                                (b'>', i) => {
                                    let to = if self.get(i - 1)? == b'/' {
                                        self.state = State::AtImmediateTagEnd;
                                        i - 1
                                    } else {
                                        self.state = State::AtTagStartDone;
                                        i
                                    };

                                    let from = self.last_index + 1;
                                    self.last_index = i;
                                    return Ok(Event::StartTag(DeferredString::new(self.get_slice(from, to))));
                                }

                                // Any other control character at this point is malformed XML!
                                (c, i) => return Err(ParseError::MalformedXML {
                                    byte: Some(i),
                                    character: Some(c),
                                    kind: MalformedXMLKind::BadTagName,
                                })
                            }
                        }
                    }
                }

                // We are inside a tag (somewhere after the name)
                State::InTag => {
                    match self.next_control_char()? {
                        // It ended, report it.
                        (b'>', i) => {
                            self.last_index = i;
                            self.state = State::InText;
                            if self.get(i - 1)? == b'/' {
                                return Ok(Event::EndTagImmediate);
                            } else {
                                return Ok(Event::StartTagDone);
                            }
                        }

                        // Whitespace character can mean two things...
                        (b'\t', i) | (b'\n', i) | (b'\r', i) | (b' ', i) => {
                            // ...continuous whitespace after tag name is nothing, ignore it.
                            if self.last_index == i - 1 {
                                self.last_index = i;
                                continue;
                            }

                            // ...non-continous whitespace means we skipped over some non-whitespace
                            // characters, it must be attribute name!

                            let from = self.last_index + 1;
                            let to = i;
                            self.last_index = i;
                            self.state = State::InTagAfterAttributeName;

                            // Consume whitespace up to and including the '=' character.
                            loop {
                                match self.next_control_char()? {
                                    (b'=', i) => {
                                        self.last_index = i;
                                        break;
                                    }

                                    (b'\t', _) | (b'\n', _) | (b'\r', _) | (b' ', _) => {
                                        // NOOP
                                    }

                                    (c, i) => {
                                        self.last_index = i;
                                        self.state = State::InTag;
                                        return Err(ParseError::MalformedXML {
                                            byte: Some(i),
                                            character: Some(c),
                                            kind: MalformedXMLKind::BadAttributeName,
                                        })
                                    }
                                }
                            }

                            return Ok(Event::AttributeName(DeferredString::new(self.get_slice(from, to))));
                        }

                        // Equal sight means we must be behind attribute name
                        (b'=', i) => {
                            let from = self.last_index + 1;
                            self.last_index = i;
                            self.state = State::InTagAfterAttributeName;
                            return Ok(Event::AttributeName(DeferredString::new(self.get_slice(from, i))));
                        }

                        // EOF is not allowed in the middle of a tag.
                        (b'\0', _) => return Err(ParseError::MalformedXML {
                            byte: None,
                            character: None,
                            kind: MalformedXMLKind::UnexpectedEof,
                        }),

                        // Other control characters are not allowed in the middle of a tag.
                        (c, i) => {
                            self.last_index = i;
                            self.state = State::InTag;
                            return Err(ParseError::MalformedXML {
                                byte: Some(i),
                                character: Some(c),
                                kind: MalformedXMLKind::BadAttributeName,
                            })
                        }
                    }
                }

                // We just saw an attribute name up to the '=' character.
                State::InTagAfterAttributeName => {
                    // Find the opening quote and figure out whether it is single or double quote
                    let (from, single_quoted) = loop {
                        match self.next_control_char()? {
                            (b'\t', _) | (b'\n', _) | (b'\r', _) | (b' ', _) => {
                                // NOOP
                            }

                            (b'"', i) => {
                                break (i + 1, false);
                            }

                            (b'\'', i) => {
                                break (i + 1, true);
                            }

                            (b'\0', _) => return Err(ParseError::MalformedXML {
                                byte: None,
                                character: None,
                                kind: MalformedXMLKind::UnexpectedEof,
                            }),

                            (c, i) => {
                                self.last_index = i;
                                self.state = State::InTag;
                                return Err(ParseError::MalformedXML {
                                    byte: Some(i),
                                    character: Some(c),
                                    kind: MalformedXMLKind::BadAttributeValue,
                                })
                            }
                        }
                    };

                    // Find the matching end quote and note if we need to decode or normalize EOLs.
                    let mut needs_decoding = false;
                    let mut needs_eol_normalizing = false;

                    let to = loop {
                        match self.next_control_char()? {
                            (b'\t', _) | (b'\n', _) | (b' ', _) | (b'=', _) | (b'>', _) => {
                                // NOOP
                            }

                            (b'\r', _) => {
                                needs_eol_normalizing = true;
                            }

                            (b'&', _) => {
                                needs_decoding = true;
                            }

                            (b'"', i) if !single_quoted => {
                                break i;
                            }

                            (b'\'', i) if single_quoted => {
                                break i;
                            }

                            (b'\0', _) => return Err(ParseError::MalformedXML {
                                byte: None,
                                character: None,
                                kind: MalformedXMLKind::UnexpectedEof,
                            }),

                            (c @ b'<', i) => return Err(ParseError::MalformedXML {
                                byte: Some(i),
                                character: Some(c),
                                kind: MalformedXMLKind::UnescapedLessThan,
                            }),

                            _ => unreachable!("Some non-control characters sneaked in!")
                        }
                    };

                    self.last_index = to;
                    self.state = State::InTag;
                    return Ok(Event::AttributeValue(DeferredString::with_options(
                        self.get_slice(from, to),
                        needs_decoding,
                        needs_eol_normalizing,
                    )));

                }

                // We are at the end of an opening tag, just report it and move to next state
                State::AtTagStartDone => {
                    self.state = State::InText;
                    return Ok(Event::StartTagDone);
                }

                // We are at the end of an opening tag that is immediately closed, just report it
                // and move to next state
                State::AtImmediateTagEnd => {
                    self.state = State::InText;
                    return Ok(Event::EndTagImmediate);
                }
            }
        }
    }

    /// Consume events until we leave given `depth` of tags. All attributes and nested tags are
    /// ignored.
    ///
    /// With depth = 0, it does nothing.
    /// With depth = 1, it finishes the current tag.
    /// With depth = 2, it finishes the current tag and its parent.
    /// ...
    pub fn finish_tag(&mut self, mut depth: usize) -> Result<(), ParseError> {
        while depth > 0 {
            match self.next()? {
                Event::StartTag(_) => depth += 1,
                Event::EndTag(_) | Event::EndTagImmediate => depth -= 1,
                Event::StartTagDone | Event::AttributeName(_) | Event::AttributeValue(_) | Event::Text(_) => { /*NOOP*/ },
                Event::Eof => return Err(ParseError::MalformedXML {
                    kind: MalformedXMLKind::UnexpectedEof,
                    byte: None, character: None,
                }),
            }
        }

        Ok(())
    }

    /// Returns the byte index of the last read character
    pub fn last_position(&self) -> usize {
        self.last_index
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;

    fn run_all_classify(input: &[u8], start: usize, positions: &mut Vec<usize>) {
        classify_fallback(input, start, positions);

        let mut positions_alt = Vec::new();

        if cfg!(target_arch = "x86_64") {
            if is_x86_feature_detected!("ssse3") {
                positions_alt.clear();
                unsafe { classify_ssse3(input, start, &mut positions_alt); }
                assert_eq!(positions, &positions_alt);
            }
            if is_x86_feature_detected!("avx2") {
                positions_alt.clear();
                unsafe { classify_ssse3(input, start, &mut positions_alt); }
                assert_eq!(positions, &positions_alt);
            }
        }
    }

    #[test]
    fn test_classify() {
        let mut input = SliceDeque::from(&b"a<b=c>d e\"f'g---------------------------------------------------"[..]);
        let mut output = Vec::new();

        run_all_classify(input.as_slice(), 0, &mut output);

        assert_eq!(output, &[
            1, 3, 5, 7, 9, 11,
        ]);
        output.clear();

        for c in 0..=255u8 {
            input.clear();
            input.extend(std::iter::repeat(c).take(BLOCK_SIZE));

            run_all_classify(input.as_slice(), 0, &mut output);

            if b"\x09\x0A\x0D \"'<=>&".contains(&c) {
                let expected_output = (0..BLOCK_SIZE).collect::<Vec<_>>();

                assert_eq!(output, expected_output);
            } else {
                assert_eq!(output, &[]);
            }

            output.clear();
        }
    }

    #[test]
    fn test_parser() {
        let xmls = [
            "<aaa bbb=\"ccc\" ddd='eee'><ggg hhh='iii'/><jjj/><kkk>lll lll</kkk><mmm>nnn</mmm></aaa>",
            "  <aaa  bbb = \"ccc\"  ddd = 'eee' > <ggg  hhh = 'iii' /> <jjj /> <kkk > lll lll </kkk > <mmm > nnn </mmm > </aaa > ",
            "<?xml bla ?><aaa bbb=\"ccc\" ddd='eee'><ggg hhh='iii'/><jjj/><kkk>lll lll</kkk><mmm>nnn</mmm></aaa>",
            "<?xml bla ?>  <aaa   bbb  =  \"ccc\"   ddd  =  'eee'  >  <ggg   hhh  =  'iii'  />  <jjj  />  <kkk  >  lll lll  </kkk  >  <mmm  >  nnn  </mmm  >  </aaa  >  ",
        ];

        for xml in &xmls {
            let mut parser = Parser::new(Cursor::new(xml));
            assert_eq!(parser.next().unwrap(), Event::StartTag(DeferredString::new(b"aaa")));
            assert_eq!(parser.next().unwrap(), Event::AttributeName(DeferredString::new(b"bbb")));
            assert_eq!(parser.next().unwrap(), Event::AttributeValue(DeferredString::new(b"ccc")));
            assert_eq!(parser.next().unwrap(), Event::AttributeName(DeferredString::new(b"ddd")));
            assert_eq!(parser.next().unwrap(), Event::AttributeValue(DeferredString::new(b"eee")));
            assert_eq!(parser.next().unwrap(), Event::StartTagDone);
            assert_eq!(parser.next().unwrap(), Event::StartTag(DeferredString::new(b"ggg")));
            assert_eq!(parser.next().unwrap(), Event::AttributeName(DeferredString::new(b"hhh")));
            assert_eq!(parser.next().unwrap(), Event::AttributeValue(DeferredString::new(b"iii")));
            assert_eq!(parser.next().unwrap(), Event::EndTagImmediate);
            assert_eq!(parser.next().unwrap(), Event::StartTag(DeferredString::new(b"jjj")));
            assert_eq!(parser.next().unwrap(), Event::EndTagImmediate);
            assert_eq!(parser.next().unwrap(), Event::StartTag(DeferredString::new(b"kkk")));
            assert_eq!(parser.next().unwrap(), Event::StartTagDone);
            assert_eq!(parser.next().unwrap(), Event::Text(DeferredString::new(b"lll lll")));
            assert_eq!(parser.next().unwrap(), Event::EndTag(DeferredString::new(b"kkk")));
            assert_eq!(parser.next().unwrap(), Event::StartTag(DeferredString::new(b"mmm")));
            assert_eq!(parser.next().unwrap(), Event::StartTagDone);
            assert_eq!(parser.next().unwrap(), Event::Text(DeferredString::new(b"nnn")));
            assert_eq!(parser.next().unwrap(), Event::EndTag(DeferredString::new(b"mmm")));
            assert_eq!(parser.next().unwrap(), Event::EndTag(DeferredString::new(b"aaa")));
            assert_eq!(parser.next().unwrap(), Event::Eof);
        }
    }

    #[test]
    fn test_parser_long_input() {
        const COUNT: usize = 1_000_000;

        let mut xml = "<aaa>".to_string();
        for _ in 0..COUNT {
            xml.push_str("<bbb/>");
        }
        xml.push_str("</aaa>");

        let mut parser = Parser::new(Cursor::new(xml));

        assert_eq!(parser.next().unwrap(), Event::StartTag(DeferredString::new(b"aaa")));
        assert_eq!(parser.next().unwrap(), Event::StartTagDone);
        for _ in 0..COUNT {
            assert_eq!(parser.next().unwrap(), Event::StartTag(DeferredString::new(b"bbb")));
            assert_eq!(parser.next().unwrap(), Event::EndTagImmediate);
        }
        assert_eq!(parser.next().unwrap(), Event::EndTag(DeferredString::new(b"aaa")));
    }
}

#[cfg(test)]
#[cfg(feature = "bencher")]
mod bench {
    use test::{Bencher, black_box};

    use super::*;

    const SAMPLE_XML: &[u8] = br#"<ShxQaSjyoqOykbi wjikp:wpf="eqqm://vvv.v6.lod/7998/XLKSrebjy-fkpqykrb" AlrsjbkqCobyqflkAyqb="7985-95-69T77:18:91" AlrsjbkqVbopflk="1" OykbiGA="81" STHKldCobyqflkAyqb="7985-95-69T82:83:82" TuGA="9" wjikp="sok:kap:axk:yjp:HphxH-QaSjyoq:u8" wpf:prebjyKlryqflk="sok:kap:axk:yjp:HphxH-QaSjyoq:u8 /ymmp/yjp/QLS/obibypb8/vbtymmp/yjp/WBH-GMD/yjpXjiSrebjy.wpa"><Sstproftbo SstproftboGA="09666799" AbufrbGA="STH_67C9959277564552"><buCeykdbVfbv BubkqTfjb="7985-95-69T82:76:99"><NofdfkyiMbqvlohGA>7</NofdfkyiMbqvlohGA><ToykpmloqSqobyjGA>7961</ToykpmloqSqobyjGA><SGSboufrbGA>2792</SGSboufrbGA><VfablOiyxfkd>qosb</VfablOiyxfkd><VfablToyrhTyd>711</VfablToyrhTyd><QsaflToyrhTyd>711</QsaflToyrhTyd><AyqyToyrhTyd>9</AyqyToyrhTyd><OolasrboGA>9</OolasrboGA><QmmifryqflkGA>9</QmmifryqflkGA><TskboGA>8</TskboGA></buCeykdbVfbv><buSsoc BubkqTfjb="7985-95-69T83:05:93"/><buCeykdbVfbv BubkqTfjb="7985-95-69T83:05:94"><NofdfkyiMbqvlohGA>7</NofdfkyiMbqvlohGA><ToykpmloqSqobyjGA>7926</ToykpmloqSqobyjGA><SGSboufrbGA>79309</SGSboufrbGA><VfablOiyxfkd>qosb</VfablOiyxfkd><VfablToyrhTyd>711</VfablToyrhTyd><QsaflToyrhTyd>711</QsaflToyrhTyd><AyqyToyrhTyd>9</AyqyToyrhTyd><OolasrboGA>9</OolasrboGA><QmmifryqflkGA>9</QmmifryqflkGA><TskboGA>8</TskboGA></buCeykdbVfbv><buCeykdbVfbv BubkqTfjb="7985-95-69T84:63:17"><NofdfkyiMbqvlohGA>7</NofdfkyiMbqvlohGA><ToykpmloqSqobyjGA>7926</ToykpmloqSqobyjGA><SGSboufrbGA>79309</SGSboufrbGA><VfablOiyxfkd>qosb</VfablOiyxfkd><VfablToyrhTyd>711</VfablToyrhTyd><QsaflToyrhTyd>711</QsaflToyrhTyd><AyqyToyrhTyd>9</AyqyToyrhTyd><OolasrboGA>9</OolasrboGA><QmmifryqflkGA>9</QmmifryqflkGA><TskboGA>8</TskboGA></buCeykdbVfbv><buSqykatxGk BubkqTfjb="7985-95-69T84:63:12"/><buSqykatxNsq BubkqTfjb="7985-95-69T84:64:98"/><buCeykdbVfbv BubkqTfjb="7985-95-69T84:64:96"><NofdfkyiMbqvlohGA>7</NofdfkyiMbqvlohGA><ToykpmloqSqobyjGA>7926</ToykpmloqSqobyjGA><SGSboufrbGA>79309</SGSboufrbGA><VfablOiyxfkd>qosb</VfablOiyxfkd><VfablToyrhTyd>711</VfablToyrhTyd><QsaflToyrhTyd>711</QsaflToyrhTyd><AyqyToyrhTyd>9</AyqyToyrhTyd><OolasrboGA>9</OolasrboGA><QmmifryqflkGA>9</QmmifryqflkGA><TskboGA>8</TskboGA></buCeykdbVfbv><buSqykatxGk BubkqTfjb="7985-95-69T84:64:81"/><buSqykatxNsq BubkqTfjb="7985-95-69T84:64:82"/><buCeykdbVfbv BubkqTfjb="7985-95-69T84:64:84"><NofdfkyiMbqvlohGA>7</NofdfkyiMbqvlohGA><ToykpmloqSqobyjGA>7975</ToykpmloqSqobyjGA><SGSboufrbGA>6575</SGSboufrbGA><VfablOiyxfkd>qosb</VfablOiyxfkd><VfablToyrhTyd>711</VfablToyrhTyd><QsaflToyrhTyd>711</QsaflToyrhTyd><AyqyToyrhTyd>9</AyqyToyrhTyd><OolasrboGA>9</OolasrboGA><QmmifryqflkGA>9</QmmifryqflkGA><TskboGA>7</TskboGA></buCeykdbVfbv><buSqykatxGk BubkqTfjb="7985-95-69T84:64:85"/><buSqykatxNsq BubkqTfjb="7985-95-69T84:64:76"/><buCeykdbVfbv BubkqTfjb="7985-95-69T84:64:71"><NofdfkyiMbqvlohGA>7</NofdfkyiMbqvlohGA><ToykpmloqSqobyjGA>7975</ToykpmloqSqobyjGA><SGSboufrbGA>6575</SGSboufrbGA><VfablOiyxfkd>qosb</VfablOiyxfkd><VfablToyrhTyd>711</VfablToyrhTyd><QsaflToyrhTyd>711</QsaflToyrhTyd><AyqyToyrhTyd>9</AyqyToyrhTyd><OolasrboGA>9</OolasrboGA><QmmifryqflkGA>9</QmmifryqflkGA><TskboGA>7</TskboGA></buCeykdbVfbv><buSqykatxGk BubkqTfjb="7985-95-69T84:64:66"/><buSqykatxNsq BubkqTfjb="7985-95-69T84:64:62"/><buSqykatxGk BubkqTfjb="7985-95-69T84:64:62"/><buSqykatxNsq BubkqTfjb="7985-95-69T84:64:06"/><buCeykdbVfbv BubkqTfjb="7985-95-69T84:64:01"><NofdfkyiMbqvlohGA>7</NofdfkyiMbqvlohGA><ToykpmloqSqobyjGA>7975</ToykpmloqSqobyjGA><SGSboufrbGA>6575</SGSboufrbGA><VfablOiyxfkd>qosb</VfablOiyxfkd><VfablToyrhTyd>711</VfablToyrhTyd><QsaflToyrhTyd>711</QsaflToyrhTyd><AyqyToyrhTyd>9</AyqyToyrhTyd><OolasrboGA>9</OolasrboGA><QmmifryqflkGA>9</QmmifryqflkGA><TskboGA>7</TskboGA></buCeykdbVfbv><buSqykatxGk BubkqTfjb="7985-95-69T84:64:19"/><buSqykatxNsq BubkqTfjb="7985-95-69T84:64:18"/><buSqykatxGk BubkqTfjb="7985-95-69T84:64:18"/><buSqykatxNsq BubkqTfjb="7985-95-69T84:64:12"/></Sstproftbo></ShxQaSjyoqOykbi>      "#;

    fn bench_classify_fn(b: &mut Bencher, classify_fn: unsafe fn(input: &[u8], start: usize, positions: &mut Vec<usize>)) {
        let input = SliceDeque::from(&SAMPLE_XML[..]);
        let mut output = Vec::new();

        b.iter(move || {
            unsafe {
                for start in (0..input.len()).step_by(BLOCK_SIZE) {
                    classify_fn(input.as_slice(), start, &mut output);
                }
            }
            black_box(&mut output).clear();
        });
    }

    #[bench]
    fn bench_classify_fallback(b: &mut Bencher) {
        bench_classify_fn(b, classify_fallback);
    }

    #[bench]
    fn bench_classify_ssse3(b: &mut Bencher) {
        if is_x86_feature_detected!("ssse3") {
            bench_classify_fn(b, classify_ssse3);
        }
    }

    /*#[bench]
    fn bench_classify_avx2(b: &mut Bencher) {
        if is_x86_feature_detected!("avx2") {
            bench_classify_fn(b, classify_avx2);
        }
    }*/
}
