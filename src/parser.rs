//! Contains low-level XML `Parser`.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::collections::VecDeque;
use std::convert::TryFrom;
use std::fmt::Display;
use std::intrinsics::transmute;
use std::io::Read;
use std::ops::Range;
use std::str::Utf8Error;

use multiversion::multiversion;
use num_enum::TryFromPrimitive;
use slice_deque::SliceDeque;

/// Amount of bytes that we attempt to read from underlying Reader
const READ_SIZE: usize = 4 * 4096; // TODO: Fine-tune.

/// We classify bytes in blocks of this size
const BLOCK_SIZE: usize = 64; // Size of u64, 64 characters, 4 sse2 128i loads, 2 avx 256i loads.

// Our custom codes for important XML characters / character groups
const CH_OTHER: u8           = 0x00; // Must be 0 because of how the SIMD algorithm works.
const CH_OTHER_UTF8: u8      = 0x80; // Same as CH_OTHER, just with extra flag
const CH_OTHER_AMPERSAND: u8 = 0x40; // Same as CH_OTHER, just with extra flag
const CH_DOUBLE_QUOTE: u8    = 0x01;
const CH_SINGLE_QUOTE: u8    = 0x02;
const CH_WHITESPACE: u8      = 0x03;
const CH_EXCL_QUEST_MARK: u8 = 0x04;
const CH_SLASH: u8           = 0x05;
const CH_LESS_THAN: u8       = 0x06;
const CH_EQUAL: u8           = 0x07;
const CH_GREATER_THAN: u8    = 0x08;


/// Naive mapping of ascii character to our code
fn char_to_code(c: u8) -> u8 {
    match c {
        b'\t' | b'\n' | b'\r' | b' ' => CH_WHITESPACE,
        b'!' | b'?' => CH_EXCL_QUEST_MARK,
        b'\"' => CH_DOUBLE_QUOTE,
        b'\'' => CH_SINGLE_QUOTE,
        b'&' => CH_OTHER_AMPERSAND,
        b'/' => CH_SLASH,
        b'<' => CH_LESS_THAN,
        b'=' => CH_EQUAL,
        b'>' => CH_GREATER_THAN,
        128..=255 => CH_OTHER_UTF8,
        _ => CH_OTHER,
    }
}

/// Fills the `chars` and `positions` vectors with codes and indexes of control characters.
///
/// If there was anything in the `chars` or `positions` vector, the new values are appended behind.
fn classify_fallback(input: &[u8], chars: &mut Vec<u8>, positions: &mut Vec<usize>) {
    let mut prev = CH_LESS_THAN; // Any that is not CH_OTHER

    for (i, c) in input.iter().enumerate() {
        let ch = char_to_code(*c);

        if ch != CH_OTHER || prev != CH_OTHER {
            chars.push(ch);
            positions.push(i);
            prev = ch;
        }
    }
}

/// Same as `classify_fallback`, but implemented using SIMD intrinsics.
///
/// # Safety
///
/// Can be only called if SSSE3 is available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "ssse3")]
unsafe fn classify_ssse3(input: &[u8], chars: &mut Vec<u8>, positions: &mut Vec<usize>) {
    // SIMD classification using two `pshufb` instructions (`_mm_shuffle_epi8` intrinsic) that match
    // the high and low nibble of the byte. Combining that together we get a non-zero value for every
    // of the matched characters.

    //     hi / lo
    //                   groups                                      characters
    //         +--------------------------------         +--------------------------------
    //         | 0 1 2 3 4 5 6 7 8 9 a b c d e f         | 0 1 2 3 4 5 6 7 8 9 a b c d e f
    //       --+--------------------------------       --+--------------------------------
    //       0 | . . . . . . . . . A A . . A . .       0 | . . . . . . . . . S S . . S . .
    //       1 | . . . . . . . . . . . . . . . .       1 | . . . . . . . . . . . . . . . .
    //       2 | B B B . . . C C . . . . . . . D       2 | S ! " . . . & ' . . . . . . . /
    //       3 | . . . . . . . . . . . . E E E F       3 | . . . . . . . . . . . . < = > ?
    //       4 | . . . . . . . . . . . . . . . .       4 | . . . . . . . . . . . . . . . .
    //       5 | . . . . . . . . . . . . . . . .       5 | . . . . . . . . . . . . . . . .
    //       6 | . . . . . . . . . . . . . . . .       6 | . . . . . . . . . . . . . . . .
    //       7 | . . . . . . . . . . . . . . . .       7 | . . . . . . . . . . . . . . . .
    //       8 | G G G G G G G G G G G G G G G G       8 | U U U U U U U U U U U U U U U U
    //       9 | G G G G G G G G G G G G G G G G       9 | U U U U U U U U U U U U U U U U
    //       a | G G G G G G G G G G G G G G G G       a | U U U U U U U U U U U U U U U U
    //       b | G G G G G G G G G G G G G G G G       b | U U U U U U U U U U U U U U U U
    //       c | G G G G G G G G G G G G G G G G       c | U U U U U U U U U U U U U U U U
    //       d | G G G G G G G G G G G G G G G G       d | U U U U U U U U U U U U U U U U
    //       e | G G G G G G G G G G G G G G G G       e | U U U U U U U U U U U U U U U U
    //       f | G G G G G G G G G G G G G G G G       f | U U U U U U U U U U U U U U U U

    // Group constants must be a single bit each, they are carefully selected such that we can later
    // use add their value to the character's ascii code to make the last 4 bits unique.
    const NOTHING: i8 = 0;
    const GROUP_A: i8 = 0x10;
    const GROUP_B: i8 = 0x20;
    const GROUP_C: i8 = 0x02;
    const GROUP_D: i8 = 0x06; // We ran out of bits, but this one may use GROUP_C|GROUP_F, because it sits exactly in position where they cross.
    const GROUP_E: i8 = 0x40;
    const GROUP_F: i8 = 0x04;
    const GROUP_G: i8 = 0x08;

    let lo_nibbles_lookup = _mm_setr_epi8(
        /* 0 */ GROUP_G | GROUP_B,
        /* 1 */ GROUP_G | GROUP_B,
        /* 2 */ GROUP_G | GROUP_B,
        /* 3 */ GROUP_G,
        /* 4 */ GROUP_G,
        /* 5 */ GROUP_G,
        /* 6 */ GROUP_G | GROUP_C,
        /* 7 */ GROUP_G | GROUP_C,
        /* 8 */ GROUP_G,
        /* 9 */ GROUP_G | GROUP_A,
        /* a */ GROUP_G | GROUP_A,
        /* b */ GROUP_G,
        /* c */ GROUP_G | GROUP_E,
        /* d */ GROUP_G | GROUP_A | GROUP_E,
        /* e */ GROUP_G | GROUP_E,
        /* f */ GROUP_G | GROUP_D | GROUP_F,
    );

    let hi_nibbles_lookup = _mm_setr_epi8(
        /* 0 */ GROUP_A,
        /* 1 */ NOTHING,
        /* 2 */ GROUP_B | GROUP_C | GROUP_D,
        /* 3 */ GROUP_E | GROUP_F,
        /* 4 */ NOTHING,
        /* 5 */ NOTHING,
        /* 6 */ NOTHING,
        /* 7 */ NOTHING,
        /* 8 */ GROUP_G,
        /* 9 */ GROUP_G,
        /* a */ GROUP_G,
        /* b */ GROUP_G,
        /* c */ GROUP_G,
        /* d */ GROUP_G,
        /* e */ GROUP_G,
        /* f */ GROUP_G,
    );

    let vec_x00 = _mm_set1_epi8(0x00);
    let vec_x0f = _mm_set1_epi8(0x0f);
    let vec_x20 = _mm_set1_epi8(0x20);
    let vec_x80 = _mm_set1_epi8(-128);

    let compact_lookup = _mm_setr_epi8(
        /* 0 */ CH_WHITESPACE as i8,
        /* 1 */ CH_EXCL_QUEST_MARK as i8,
        /* 2 */ CH_DOUBLE_QUOTE as i8,
        /* 3 */ -1, // Should not be present
        /* 4 */ CH_OTHER_AMPERSAND as i8,
        /* 5 */ CH_SINGLE_QUOTE as i8,
        /* 6 */ -1, // Should not be present
        /* 7 */ -1, // Should not be present
        /* 8 */ CH_OTHER_UTF8 as i8,
        /* 9 */ CH_SLASH as i8,
        /* a */ -1, // Should not be present
        /* b */ CH_EXCL_QUEST_MARK as i8,
        /* c */ CH_LESS_THAN as i8,
        /* d */ CH_EQUAL as i8,
        /* e */ CH_GREATER_THAN as i8,
        /* f */ CH_OTHER as i8, // This doesn't actually matter because whitespaces have the highest bit set and that's why it is mapped to 0, not because of this.
    );

    // Get the `input` as raw pointers and assert that both start and end are aligned to `BLOCK_SIZE`
    let mut ptr = input.as_ptr();
    let end = input.as_ptr().add(input.len());
    debug_assert!(ptr as usize % BLOCK_SIZE == 0);
    debug_assert!(end as usize % BLOCK_SIZE == 0);

    let mut offset = 0;

    #[repr(align(64))]
    struct ScratchPad([u8; BLOCK_SIZE]);
    let mut scratchpad = ScratchPad([0; 64]);

    // We don't want to keep state between runs of the classifier, so here we just assume that the
    // previous block ended in a control character, so we should emit OTHER_CHAR if we don't begin
    // with control character. If it is not true, we will have two OTHER_CHARs in a row, which is
    // not a big deal, the state machine can handle that.
    let mut prev_mask = 1u64 << 63;

    while ptr < end {
        let load_vec = |ptr: *const u8, out: *mut u8| -> u64 {
            // Load input from memory to SIMD register
            //
            //   Example: input = ['A', '\t', '\n', '\r', ' ', '!', '"', '\'', '/', '<', '=', '>', '?', '\xea', ..]
            //            input = [41, 09, 0a, 0d, 20, 21, 22, 26, 27, 2f, 3c, 3d, 3e, 3f, ea, ..]
            let input = _mm_load_si128(ptr as *const __m128i); // TODO: Stream load could be good, but it is not available in rust. :-(

            // Map characters into groups using two shuffles
            //
            //   Example: groups = [00, 10, 10, 10, 20, 20, 20, 02, 02, 06, 40, 40, 40, 04, 08, ...]
            let lo_nibbles = _mm_and_si128(input, vec_x0f);
            let hi_nibbles = _mm_and_si128(_mm_srli_epi16(input, 4), vec_x0f);
            let lo_translated = _mm_shuffle_epi8(lo_nibbles_lookup, lo_nibbles);
            let hi_translated = _mm_shuffle_epi8(hi_nibbles_lookup, hi_nibbles);
            let groups = _mm_and_si128(lo_translated, hi_translated);

            // Test for zeroes (i.e. the "other" characters)
            //
            //   Example: mask = [ff, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, ...]
            let mask = _mm_cmpeq_epi8(groups, vec_x00);

            // Squash all characters above 0x80 to 0x80 (Utf-8 characters)
            //
            //   Example: input = [41, 09, 0a, 0d, 20, 21, 22, 26, 27, 2f, 3c, 3d, 3e, 3f, 80, ..]
            let input = _mm_min_epu8(input, vec_x80);

            // Turn all non-matched characters to 0xff
            //
            //   Example: input = [ff, 09, 0a, 0d, 20, 21, 22, 26, 27, 2f, 3c, 3d, 3e, 3f, 80, ..]
            let input = _mm_or_si128(input, mask);

            // Squash all characters <= 0x20 to 0 (now whitespace characters only) using saturating
            // subtraction.
            //
            //   Example: input = [df, 00, 00, 00, 00, 01, 02, 06, 07, 0f, 1c, 1d, 1e, 1f, 60, ..]
            let input = _mm_subs_epu8(input, vec_x20);

            // Xor groups into the input. Thanks to carefully selected group numbers, this will
            // make sure that every character of interest gets unique last 4 bits.
            //
            //   Example: input = [df, 10, 10, 10, 20, 21, 22, 04, 05, 09, 5c, 5d, 5e, 1b, 68, ..]
            //     Only last 4 bits really matter, unless the highest bit is set
            //                    [df, _0, _0, _0, _0, _1, _2, _4, _5, _9, _c, _d, _e, _b, _8, ..]
            let input = _mm_xor_si128(input, groups);

            // Now we have unique last 4 bits and the highest bit is not set for any but CH_OTHER.
            // We use another shuffle to map these values (which are all over the place in the 0..F
            // range) into a compacted list.
            // Note that we even have distinct value for '!' and '?', but we map them into the same
            // value here to save space in the DFA.
            //
            //   Example: input = [00, 01, 01, 01, 01, 02, 03, 04, 06, 07, 08, 09, 02, 0a, ..]
            let input = _mm_shuffle_epi8(compact_lookup, input);

            // Store it to our properly aligned scratchpad
            _mm_store_si128(out as *mut __m128i, input);

            // Return the mask as bit mask
            (_mm_movemask_epi8(mask) as u16) as u64
        };

        let mask_a = load_vec(ptr,         &mut scratchpad.0[0]  as *mut u8);
        let mask_b = load_vec(ptr.add(16), &mut scratchpad.0[16] as *mut u8);
        let mask_c = load_vec(ptr.add(32), &mut scratchpad.0[32] as *mut u8);
        let mask_d = load_vec(ptr.add(48), &mut scratchpad.0[48] as *mut u8);

        let mut mask = !(mask_a | (mask_b << 16) | (mask_c << 32) | (mask_d << 48));

        // Shift the mask such that we take one non-control character after each block of control
        // characters. Remember the mask for next time.
        let tmp = mask;
        mask = mask | (mask << 1) | (prev_mask >> 63);
        prev_mask = tmp;

        // If we found any control characters in this block
        if mask != 0 {
            let count = mask.count_ones() as usize;

            // Reserve enough space in case the whole block was filled with control characters
            chars.reserve(BLOCK_SIZE); // Max amount of control characters we may find.
            positions.reserve(BLOCK_SIZE); // Max amount of control characters we may find.
            let mut chars_ptr = chars.as_mut_ptr().offset(chars.len() as isize);
            let mut positions_ptr = positions.as_mut_ptr().offset(positions.len() as isize);

            // Write them out.
            loop {
                // To reduce the amount of conditional jumps, we always write out the next 4
                // characters, even if there is less than 4 characters left. In that case, some of
                // them are incorrect, but they will be ignored because at the end we set the length
                // to the actual total amount of found characters.

                // TODO: Verify that this is unrolled
                // TODO: Fine-tune the amount of repetitions
                for _ in 0..4 {
                    let index = mask.trailing_zeros() as usize;

                    let ch = *scratchpad.0.get_unchecked(index); // Safety: u64 has max 63 trailing zeroes, so that will never overflow
                    let pos = index + offset;

                    *chars_ptr = ch;
                    chars_ptr = chars_ptr.add(1);
                    *positions_ptr = pos;
                    positions_ptr = positions_ptr.add(1);

                    mask = mask & mask.overflowing_sub(1).0;
                }

                if mask == 0 {
                    break;
                }
            }

            chars.set_len(chars.len() + count);
            positions.set_len(positions.len() + count);
        }

        ptr = ptr.offset(BLOCK_SIZE as isize);
        offset += BLOCK_SIZE;
    }
}

/// Same as `classify_ssse3`, but using AVX2 instructions.
///
/// # Safety
///
/// Can be only called if AVX2 is available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn classify_avx2(input: &[u8], chars: &mut Vec<u8>, positions: &mut Vec<usize>) {
    // Same algorithm as classify_ssse3, but working on half the amount of twice as long vectors.

    const NOTHING: i8 = 0;
    const GROUP_A: i8 = 0x10;
    const GROUP_B: i8 = 0x20;
    const GROUP_C: i8 = 0x02;
    const GROUP_D: i8 = 0x06;
    const GROUP_E: i8 = 0x40;
    const GROUP_F: i8 = 0x04;
    const GROUP_G: i8 = 0x08;

    let lo_nibbles_lookup = _mm256_setr_epi8(
        /* 0 */ GROUP_G | GROUP_B,
        /* 1 */ GROUP_G | GROUP_B,
        /* 2 */ GROUP_G | GROUP_B,
        /* 3 */ GROUP_G,
        /* 4 */ GROUP_G,
        /* 5 */ GROUP_G,
        /* 6 */ GROUP_G | GROUP_C,
        /* 7 */ GROUP_G | GROUP_C,
        /* 8 */ GROUP_G,
        /* 9 */ GROUP_G | GROUP_A,
        /* a */ GROUP_G | GROUP_A,
        /* b */ GROUP_G,
        /* c */ GROUP_G | GROUP_E,
        /* d */ GROUP_G | GROUP_A | GROUP_E,
        /* e */ GROUP_G | GROUP_E,
        /* f */ GROUP_G | GROUP_D | GROUP_F,

        /* 0 */ GROUP_G | GROUP_B,
        /* 1 */ GROUP_G | GROUP_B,
        /* 2 */ GROUP_G | GROUP_B,
        /* 3 */ GROUP_G,
        /* 4 */ GROUP_G,
        /* 5 */ GROUP_G,
        /* 6 */ GROUP_G | GROUP_C,
        /* 7 */ GROUP_G | GROUP_C,
        /* 8 */ GROUP_G,
        /* 9 */ GROUP_G | GROUP_A,
        /* a */ GROUP_G | GROUP_A,
        /* b */ GROUP_G,
        /* c */ GROUP_G | GROUP_E,
        /* d */ GROUP_G | GROUP_A | GROUP_E,
        /* e */ GROUP_G | GROUP_E,
        /* f */ GROUP_G | GROUP_D | GROUP_F,
    );

    let hi_nibbles_lookup = _mm256_setr_epi8(
        /* 0 */ GROUP_A,
        /* 1 */ NOTHING,
        /* 2 */ GROUP_B | GROUP_C | GROUP_D,
        /* 3 */ GROUP_E | GROUP_F,
        /* 4 */ NOTHING,
        /* 5 */ NOTHING,
        /* 6 */ NOTHING,
        /* 7 */ NOTHING,
        /* 8 */ GROUP_G,
        /* 9 */ GROUP_G,
        /* a */ GROUP_G,
        /* b */ GROUP_G,
        /* c */ GROUP_G,
        /* d */ GROUP_G,
        /* e */ GROUP_G,
        /* f */ GROUP_G,

        /* 0 */ GROUP_A,
        /* 1 */ NOTHING,
        /* 2 */ GROUP_B | GROUP_C | GROUP_D,
        /* 3 */ GROUP_E | GROUP_F,
        /* 4 */ NOTHING,
        /* 5 */ NOTHING,
        /* 6 */ NOTHING,
        /* 7 */ NOTHING,
        /* 8 */ GROUP_G,
        /* 9 */ GROUP_G,
        /* a */ GROUP_G,
        /* b */ GROUP_G,
        /* c */ GROUP_G,
        /* d */ GROUP_G,
        /* e */ GROUP_G,
        /* f */ GROUP_G,
    );

    let vec_x00 = _mm256_set1_epi8(0x00);
    let vec_x0f = _mm256_set1_epi8(0x0f);
    let vec_x20 = _mm256_set1_epi8(0x20);
    let vec_x80 = _mm256_set1_epi8(-128);

    let compact_lookup = _mm256_setr_epi8(
        /* 0 */ CH_WHITESPACE as i8,
        /* 1 */ CH_EXCL_QUEST_MARK as i8,
        /* 2 */ CH_DOUBLE_QUOTE as i8,
        /* 3 */ -1, // Should not be present
        /* 4 */ CH_OTHER_AMPERSAND as i8,
        /* 5 */ CH_SINGLE_QUOTE as i8,
        /* 6 */ -1, // Should not be present
        /* 7 */ -1, // Should not be present
        /* 8 */ CH_OTHER_UTF8 as i8,
        /* 9 */ CH_SLASH as i8,
        /* a */ -1, // Should not be present
        /* b */ CH_EXCL_QUEST_MARK as i8,
        /* c */ CH_LESS_THAN as i8,
        /* d */ CH_EQUAL as i8,
        /* e */ CH_GREATER_THAN as i8,
        /* f */ CH_OTHER as i8,

        /* 0 */ CH_WHITESPACE as i8,
        /* 1 */ CH_EXCL_QUEST_MARK as i8,
        /* 2 */ CH_DOUBLE_QUOTE as i8,
        /* 3 */ -1, // Should not be present
        /* 4 */ CH_OTHER_AMPERSAND as i8,
        /* 5 */ CH_SINGLE_QUOTE as i8,
        /* 6 */ -1, // Should not be present
        /* 7 */ -1, // Should not be present
        /* 8 */ CH_OTHER_UTF8 as i8,
        /* 9 */ CH_SLASH as i8,
        /* a */ -1, // Should not be present
        /* b */ CH_EXCL_QUEST_MARK as i8,
        /* c */ CH_LESS_THAN as i8,
        /* d */ CH_EQUAL as i8,
        /* e */ CH_GREATER_THAN as i8,
        /* f */ CH_OTHER as i8,
    );

    let mut ptr = input.as_ptr();
    let end = input.as_ptr().add(input.len());
    debug_assert!(ptr as usize % BLOCK_SIZE == 0);
    debug_assert!(end as usize % BLOCK_SIZE == 0);

    let mut offset = 0;

    #[repr(align(64))]
    struct ScratchPad([u8; BLOCK_SIZE]);
    let mut scratchpad = ScratchPad([0; 64]);

    let mut prev_mask = 1u64 << 63;

    while ptr < end {
        let mask_a = {
            let input = _mm256_load_si256(ptr as *const __m256i); // TODO: Stream load could be good, but it is not available in rust. :-(

            let lo_nibbles = _mm256_and_si256(input, vec_x0f);
            let hi_nibbles = _mm256_and_si256(_mm256_srli_epi16(input, 4), vec_x0f);
            let lo_translated = _mm256_shuffle_epi8(lo_nibbles_lookup, lo_nibbles);
            let hi_translated = _mm256_shuffle_epi8(hi_nibbles_lookup, hi_nibbles);
            let groups = _mm256_and_si256(lo_translated, hi_translated);

            let mask = _mm256_cmpeq_epi8(groups, vec_x00);

            let input = _mm256_min_epu8(input, vec_x80);

            let input = _mm256_or_si256(input, mask);

            let input = _mm256_subs_epu8(input, vec_x20);

            let input = _mm256_xor_si256(input, groups);

            let input = _mm256_shuffle_epi8(compact_lookup, input);

            _mm256_store_si256(&mut scratchpad.0[0] as *mut u8 as *mut __m256i, input);

            (_mm256_movemask_epi8(mask) as u32) as u64
        };

        let mask_b = {
            let input = _mm256_load_si256(ptr.add(32) as *const __m256i); // TODO: Stream load could be good, but it is not available in rust. :-(

            let lo_nibbles = _mm256_and_si256(input, vec_x0f);
            let hi_nibbles = _mm256_and_si256(_mm256_srli_epi16(input, 4), vec_x0f);
            let lo_translated = _mm256_shuffle_epi8(lo_nibbles_lookup, lo_nibbles);
            let hi_translated = _mm256_shuffle_epi8(hi_nibbles_lookup, hi_nibbles);
            let groups = _mm256_and_si256(lo_translated, hi_translated);

            let mask = _mm256_cmpeq_epi8(groups, vec_x00);

            let input = _mm256_min_epu8(input, vec_x80);
            let input = _mm256_or_si256(input, mask);
            let input = _mm256_subs_epu8(input, vec_x20);
            let input = _mm256_xor_si256(input, groups);
            let input = _mm256_shuffle_epi8(compact_lookup, input);

            _mm256_store_si256(&mut scratchpad.0[32] as *mut u8 as *mut __m256i, input);

            (_mm256_movemask_epi8(mask) as u32) as u64
        };

        let mut mask = !(mask_a | (mask_b << 32));

        let tmp = mask;
        mask = mask | (mask << 1) | (prev_mask >> 63);
        prev_mask = tmp;

        if mask != 0 {
            let count = mask.count_ones() as usize;

            chars.reserve(BLOCK_SIZE); // Max amount of control characters we may find.
            positions.reserve(BLOCK_SIZE); // Max amount of control characters we may find.
            let mut chars_ptr = chars.as_mut_ptr().offset(chars.len() as isize);
            let mut positions_ptr = positions.as_mut_ptr().offset(positions.len() as isize);

            loop {
                // TODO: Verify that this is unrolled
                // TODO: Fine-tune the amount of repetitions
                for _ in 0..4 {
                    let index = mask.trailing_zeros() as usize;

                    let ch = *scratchpad.0.get_unchecked(index); // Safety: u64 has max 63 trailing zeroes, so that will never overflow
                    let pos = index + offset;

                    *chars_ptr = ch;
                    chars_ptr = chars_ptr.add(1);
                    *positions_ptr = pos;
                    positions_ptr = positions_ptr.add(1);

                    mask = mask & mask.overflowing_sub(1).0;
                }

                if mask == 0 {
                    break;
                }
            }

            chars.set_len(chars.len() + count);
            positions.set_len(positions.len() + count);
        }

        ptr = ptr.offset(BLOCK_SIZE as isize);
        offset += BLOCK_SIZE;
    }
}

#[multiversion]
#[specialize(target = "[x86|x86_64]+avx2", fn = "classify_avx2", unsafe = true)]
#[specialize(target = "[x86|x86_64]+ssse3", fn = "classify_ssse3", unsafe = true)]
fn classify(input: &[u8], chars: &mut Vec<u8>, positions: &mut Vec<usize>) {
    classify_fallback(input, chars, positions)
}

/// If this bit is set in a transition, it means that event should be emitted
const BIT_EMIT: u8 = 0b1000_0000;

// If this bit is set in a transition, it means the current position should be saved as
// the start of future event (if any).
const BIT_SAVE_START: u8 = 0b0100_0000;

// If this bit is set in a transition, it means the current position should be saved as
// the end of future event (if any).
const BIT_SAVE_END: u8 = 0b0010_0000;

// This table is generated by the `build_dfa` binary. It maps from current state and control
// character into new state and flags.
const PARSE_DFA: [u8; 19 * 9] = [
/*             00    01    02    03    04    05    06    07    08 */
/*   00:  */ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
/*   01:  */ 0x01, 0x00, 0x00, 0xaa, 0x00, 0xa3, 0x00, 0x00, 0xa7,
/*   02:  */ 0x02, 0x00, 0x00, 0xab, 0x00, 0x00, 0x00, 0x00, 0xa7,
/*   03:  */ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x87,
/*   04:  */ 0x04, 0x04, 0x04, 0x2c, 0x04, 0x04, 0xa8, 0x04, 0x04,
/*   05:  */ 0x05, 0x00, 0x00, 0xad, 0x00, 0x00, 0x00, 0xaf, 0x00,
/*   06:  */ 0x06, 0xb2, 0x06, 0x06, 0x06, 0x06, 0x00, 0x06, 0x06,
/*   07:  */ 0x44, 0x44, 0x44, 0x07, 0x44, 0x44, 0x08, 0x44, 0x44,
/*   08:  */ 0x41, 0x00, 0x00, 0x00, 0x00, 0x09, 0x00, 0x00, 0x00,
/*   09:  */ 0x42, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
/*   0a:  */ 0x45, 0x00, 0x00, 0x0a, 0x00, 0x03, 0x00, 0x00, 0x07,
/*   0b:  */ 0x00, 0x00, 0x00, 0x0b, 0x00, 0x00, 0x00, 0x00, 0x07,
/*   0c:  */ 0x04, 0x04, 0x04, 0x0c, 0x04, 0x04, 0x88, 0x04, 0x04,
/*   0d:  */ 0x00, 0x00, 0x00, 0x0d, 0x00, 0x00, 0x00, 0x0f, 0x00,
/*   0e:  */ 0x0e, 0x0e, 0xb2, 0x0e, 0x0e, 0x0e, 0x00, 0x0e, 0x0e,
/*   0f:  */ 0x00, 0x10, 0x11, 0x0f, 0x00, 0x00, 0x00, 0x00, 0x00,
/*   10:  */ 0x46, 0x00, 0x46, 0x46, 0x46, 0x46, 0x00, 0x46, 0x46,
/*   11:  */ 0x4e, 0x4e, 0x00, 0x4e, 0x4e, 0x4e, 0x00, 0x4e, 0x4e,
/*   12:  */ 0x00, 0x00, 0x00, 0x0a, 0x00, 0x03, 0x00, 0x00, 0x07,
];

// If this bit is set in event code, the string of the event contains UTF-8 characters. (Actually
// non-ascii characters that must be validated as valid UTF-8.)
// The same bit is set in the code of the character that represents non-ascii character.
const BIT_HAS_UTF8: u8 =    0b1000_0000;

// If this bit is set in event code, the string of the event contains XML escapes.
// The same bit is set in the code of ampersand.
const BIT_HAS_ESCAPES: u8 = 0b0100_0000;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum EventCode {
    // 0 is never reported as enum EventCode, but it is used in u8 representation to report error.

    StartTag =        0o1,

    EndTag =          0o2,
    EndTagImmediate = 0o3,

    Text =            0o4,

    AttributeName =   0o5,
    AttributeValue =  0o6,

    Eof =             0o7,
}

// States are chosen in such way, that the lower 3 bits contain the code of the event that would be
// emitted in that state, if that state can ever emit an event.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, TryFromPrimitive)]
#[repr(u8)]
enum State {
    /// Exception is a state that is not handled by DFA. It signifies either error in XML syntax, or
    /// a XML construct that must be handled by specially and is not covered by the DFA.
    Exception = 0,

    /// State outside of any tag or text. This is the starting state.
    Outside = EventCode::Eof as u8, // emits EventCode::Eof

    TagStart = 0o10,
    TagEnd = 0o11,

    TagName = EventCode::StartTag as u8, // emits EventCode::StartTag
    TagEndName = EventCode::EndTag as u8, // emits EventCode::EndTag

    InTag = 0o12,
    InTagEnd = 0o13,

    AttrName = EventCode::AttributeName as u8, // emits EventCode::AttributeName

    AfterAttrName = 0o14, // Space between attribute name and '=' sign

    AfterAttrEq = 0o15,

    AttrValueDoubleQuotedOpen = 0o21,
    AttrValueDoubleQuoted = EventCode::AttributeValue as u8, // emits EventCode::AttributeValue

    AttrValueSingleQuotedOpen = 0o20,
    AttrValueSingleQuoted = EventCode::AttributeValue as u8 | 0o10, // emits EventCode::AttributeValue

    AfterAttrValue = 0o17,

    AfterImmediateEndTag = EventCode::EndTagImmediate as u8, // emits EventCode::EndTagImmediate

    InText = EventCode::Text as u8, // emits EventCode::Text
    InTextEndWhitespace = 0o22,

    // Exception states. These states are set by exception handler, if the buffer did not contain
    // enough data to handle the exception. (E.g. comment started in current buffer but didn't end
    // in it.)

    HandledException = 0o100,
    InProcessingInstruction = 0o101,
    InComment = 0o102,
    InCDATA = 0o103,
}

#[derive(Clone, Debug)]
struct ParserState {
    state: State,
    flags: u8,
    start_position: usize,
    end_position: usize,
    exception_up_to: usize,
}

impl Default for ParserState {
    fn default() -> Self {
        Self {
            state: State::Outside,
            flags: 0,
            start_position: 0,
            end_position: 0,
            exception_up_to: 0,
        }
    }
}

#[derive(Debug)]
struct StateMachine<'e, 'state> {
    events: &'e mut VecDeque<InternalEvent>,
    errors: &'e mut VecDeque<MalformedXmlError>,
    state: &'state mut ParserState,
    position_offset: usize,
}

impl<'e, 'state> StateMachine<'e, 'state> {
    fn run(&mut self, chars: &Vec<u8>, positions: &Vec<usize>, buffer: &[u8]) {
        // If we were inside exception handler when we ended last time, we must continue with it.
        if self.state.state as u8 > State::HandledException as u8 {
            self.continue_exception(buffer);

            if self.state.state as u8 > State::HandledException as u8 {
                // If the exception handler didn't change the state, we need more data in the buffer.
                return;
            }
        }

        let mut s = (*self.state).clone();

        // Go over all control characters and their positions
        for (ch, pos) in chars.iter().copied().zip(positions.iter().copied()) {
            let out_pos = pos + self.position_offset;

            if out_pos < s.exception_up_to {
                // We are still receiving characters that are in area that was handled by exception
                // handler.
                continue;
            }

            // First we OR-in the flags from the characters (e.g. has-utf8, has-escapes).
            // These are always additive. The lower bits from the characters get ORed too, it doesn't
            // matter.
            s.flags |= ch;

            let dfa_index = s.state as u8 * 9 + (ch & 0b1111);
            let transition = unsafe { *PARSE_DFA.get_unchecked(dfa_index as usize) };

            if transition == State::Exception as u8 {
                *self.state = s;
                self.handle_exception(out_pos, buffer);
                s = (*self.state).clone();

                if s.state as u8 > State::HandledException as u8 {
                    // If we are still in exception state after running the exception handler, then
                    // we need more data in the buffer to run the handler.
                    return;
                } else {
                    continue;
                }
            }

            if transition & BIT_SAVE_END != 0 {
                s.end_position = out_pos;
            }

            if transition & BIT_EMIT != 0 {
                let code = (s.state as u8 & 0b0000_0111) | (s.flags & 0b1100_0000);
                self.events.push_back(InternalEvent {
                    start: s.start_position,
                    end: s.end_position,
                    code,
                });
                s.flags = 0;
            }

            if transition & BIT_SAVE_START != 0 {
                s.start_position = out_pos;
            }

            s.state = match transition & 0b0001_1111 {
                #[cfg(debug_assertions)]
                state_num => State::try_from(state_num).unwrap(),

                #[cfg(not(debug_assertions))]
                state_num => unsafe { std::mem::transmute::<u8, State>(state_num) },
            };
        }

        *self.state = s;
    }

    // The DFA covers only the most usual states in the XML state machine. Anything else will fall
    // into exception state from which we have to manually recover. Exception may be either error
    // in XML syntax or some less usual XML construct.
    #[cold]
    fn handle_exception(&mut self, pos: usize, buffer: &[u8]) {
        match (self.state.state, buffer[pos]) {
            (State::TagStart, b'?') =>
                self.handle_processing_instruction(pos, buffer),

            (State::TagStart, b'!') if buffer[pos..].starts_with(b"!--") =>
                self.handle_comment(pos, buffer),

            (State::TagStart, b'!') if buffer[pos..].starts_with(b"![CDATA[") =>
                self.handle_cdata(pos, buffer),

            _ =>
                self.handle_error(pos, buffer),
        }
    }

    #[cold]
    fn continue_exception(&mut self, buffer: &[u8]) {
        match self.state.state {
            State::InProcessingInstruction =>
                self.continue_processing_instruction(buffer),

            State::InComment =>
                self.continue_comment(buffer),

            State::InCDATA =>
                self.continue_cdata(buffer),

            _ =>
                unreachable!("continue_exception was called with state {:?}", self.state.state),
        }
    }

    fn handle_processing_instruction(&mut self, pos: usize, buffer: &[u8]) {
        self.state.start_position = pos + 2;
        self.state.state = State::InProcessingInstruction;

        self.continue_processing_instruction(buffer);
    }

    fn continue_processing_instruction(&mut self, buffer: &[u8]) {
        // Currently we ignore the content and just skip at its end.
        // TODO: Should we handle them anyhow? We could at least emit their content as event...

        debug_assert_eq!(self.state.state, State::InProcessingInstruction);

        let pi_start = self.state.start_position;
        let pi_end = twoway::find_bytes(&buffer[pi_start..], b"?>");
        if let Some(pi_end) = pi_end {
            let pi_end = pi_end + pi_start;
            self.state.exception_up_to = pi_end + 2;
            self.state.state = State::Outside;
            self.state.flags = 0;
        } else {
            // The state will remain State::InProcessingInstruction, so once we have new longer buffer, we will be called again.
        }
    }

    fn handle_comment(&mut self, pos: usize, buffer: &[u8]) {
        self.state.start_position = pos + 2;
        self.state.state = State::InComment;

        self.continue_comment(buffer);
    }

    fn continue_comment(&mut self, buffer: &[u8]) {
        // Currently we skip comments and just skip at its end.
        // TODO: Should we handle them anyhow? We could at least emit their content as event...

        debug_assert_eq!(self.state.state, State::InComment);

        let comment_start = self.state.start_position;
        let comment_end = twoway::find_bytes(&buffer[comment_start..], b"-->");
        if let Some(comment_end) = comment_end {
            let comment_end = comment_end + comment_start;
            self.state.exception_up_to = comment_end + 3;
            self.state.state = State::Outside;
            self.state.flags = 0;
        } else {
            // The state will remain State::InComment, so once we have new longer buffer, we will be called again.
        }
    }

    fn handle_cdata(&mut self, pos: usize, buffer: &[u8]) {
        // TODO: Any specialities we should care about in CDATA?

        if self.state.state != State::InCDATA {
            self.state.start_position = pos + 8;
            self.state.state = State::InCDATA;
        };

        self.continue_cdata(buffer);
    }

    fn continue_cdata(&mut self, buffer: &[u8]) {
        // TODO: Any specialities we should care about in CDATA?

        debug_assert_eq!(self.state.state, State::InCDATA);

        let cdata_start = self.state.start_position;
        let cdata_end = twoway::find_bytes(&buffer[cdata_start..], b"]]>");
        if let Some(cdata_end) = cdata_end {
            let cdata_end = cdata_end + cdata_start;

            let code = EventCode::Text as u8 | BIT_HAS_UTF8; // We conservatively assume that Utf-8 characters were present.
            self.events.push_back(InternalEvent {
                start: cdata_start,
                end: cdata_end,
                code,
            });

            self.state.exception_up_to = cdata_end + 3;
            self.state.state = State::Outside;
            self.state.flags = 0;
        } else {
            // The state will remain State::InCDATA, so once we have new longer buffer, we will be called again.
        }
    }

    fn handle_error(&mut self, pos: usize, buffer: &[u8]) {
        // Push error event and corresponding event
        self.events.push_back(InternalEvent {
            start: pos,
            end: pos,
            code: 0,
        });

        let context_slice = pos.saturating_sub(10)..(pos + 10).min(buffer.len());
        self.errors.push_back(MalformedXmlError {
            kind: MalformedXMLKind::Other,
            context: Some(String::from_utf8_lossy(&buffer[context_slice]).into_owned()),
        });

        // Attempt to recover
        // We find the closest '>' and resume behind as if we just left tag. This may be incorrect,
        // the '>' character may be inside of a comment or cdata. But in that case we will just
        // report more errors until we eventually recover.
        let continue_from = twoway::find_bytes(&buffer[(pos+1)..], b">").unwrap_or(0);
        self.state.exception_up_to = pos + 1 + continue_from;
        self.state.state = State::Outside;
        self.state.flags = 0;
        // TODO: Smarter recoveries?
    }
}

/// The kind of XML malformation that was encountered
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MalformedXMLKind {
/*
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
*/

    /// EOF before the XML structure was properly terminated
    UnexpectedEof,

    /// Any other problem
    Other,
}

#[derive(Clone, Debug)]
pub struct MalformedXmlError {
    /// Kind of malformation
    kind: MalformedXMLKind,

    /// Text around the problem
    context: Option<String>,
}

impl std::fmt::Display for MalformedXmlError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(f, "Malformed XML")?;

        if self.kind != MalformedXMLKind::Other {
            let msg = match self.kind {
                /*MalformedXMLKind::BadTagName => "bad tag name",
                MalformedXMLKind::BadAttributeName => "bad attribute name",
                MalformedXMLKind::BadAttributeValue => "bad value name",
                MalformedXMLKind::ExtrasInClosingTag => "extra things in closing tag",
                MalformedXMLKind::UnescapedLessThan => "unescaped '<' character",*/
                MalformedXMLKind::UnexpectedEof => "unexpected EOF",
                MalformedXMLKind::Other => "",
            };

            write!(f, " ({})", msg)?;
        }

        if let Some(context) = &self.context {
            write!(f, ": {:?}", context)?;
        }

        Ok(())
    }
}

impl std::error::Error for MalformedXmlError {}

/// Error while parsing
#[derive(Debug)]
pub enum ParseError {
    /// IO error reading from the underlying Read
    IO(std::io::Error),

    /// XML was not well formed
    MalformedXML(MalformedXmlError),
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        match self {
            ParseError::IO(err) => Display::fmt(err, f),
            ParseError::MalformedXML(err) => Display::fmt(err, f),
        }
    }
}

impl std::error::Error for ParseError {}

impl From<std::io::Error> for ParseError {
    fn from(err: std::io::Error) -> Self {
        ParseError::IO(err)
    }
}

impl From<MalformedXmlError> for ParseError {
    fn from(err: MalformedXmlError) -> Self {
        ParseError::MalformedXML(err)
    }
}

#[derive(Debug)]
pub enum DecodeError {
    BadUtf8,
    BadEscape,
}

impl std::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        match self {
            DecodeError::BadUtf8 => write!(f, "bad utf8"),
            DecodeError::BadEscape => write!(f, "bad xml escape"),
        }
    }
}

impl std::error::Error for DecodeError {}

impl From<std::str::Utf8Error> for DecodeError {
    fn from(_err: std::str::Utf8Error) -> Self {
        DecodeError::BadUtf8
    }
}

#[derive(Debug)]
struct InternalEvent {
    start: usize,
    end: usize,
    code: u8,
}

#[derive(Debug)]
pub struct Event<'a> {
    slice: &'a mut [u8],
    code: u8,
}

impl<'a> Event<'a> {
    #[inline(always)]
    fn new(internal_event: &InternalEvent, buffer: &'a mut [u8]) -> Self {
        Self {
            slice: &mut buffer[internal_event.start..internal_event.end],
            code: internal_event.code,
        }
    }

    fn eof() -> Self {
        Self {
            slice: &mut [],
            code: EventCode::Eof as u8,
        }
    }

    #[inline(always)]
    pub fn code(&self) -> EventCode {
        // Safety: We take only last 3 bits and we know that every combination of them is a valid
        // `EventCode`
        unsafe {
            transmute(self.code & 0b00000111)
        }
    }

    #[cold]
    fn check_utf8(&self) -> Result<(), Utf8Error> {
        std::str::from_utf8(self.slice).map(|_| ())
    }

    #[cold]
    fn decode_escapes(&mut self) -> Result<(), DecodeError> {
        fn find_next_escape(slice: &[u8], from: usize) -> Option<(usize, usize, char)> {
            let start = slice[from..].iter().position(|c| *c == b'&')? + from;
            let end = slice[start..].iter().position(|c| *c == b';')? + start;

            match &slice[start+1..end] {
                b"lt"   => Some((start, end, '<')),
                b"gt"   => Some((start, end, '>')),
                b"amp"  => Some((start, end, '&')),
                b"apos" => Some((start, end, '\'')),
                b"quot" => Some((start, end, '"')),

                [b'#', b'x', hexa @ ..] => {
                    let c = std::char::from_u32(btoi::btou_radix(hexa, 16).ok()?)?;
                    Some((start, end, c))
                },

                [b'#', decimal @ ..] => {
                    let c = std::char::from_u32(btoi::btou(decimal).ok()?)?;
                    Some((start, end, c))
                },

                _ => None,
            }
        }

        let mut escape = match find_next_escape(&self.slice, 0) {
            Some(escape) => escape,
            None => return Ok(()),
        };
        let mut current_end = escape.0;

        loop {
            let next_escape = find_next_escape(&self.slice, escape.1);
            let next_escape_end = next_escape.map(|n| n.0).unwrap_or(self.slice.len());

            let utf8_len = escape.2.encode_utf8(&mut self.slice[current_end..]).len();
            debug_assert!(utf8_len <= escape.1 - escape.0, "We got XML escape that is shorter than its UTF-8 representation. That should not be possible.");
            current_end += utf8_len;

            self.slice.copy_within((escape.1 + 1)..next_escape_end, current_end);
            current_end += next_escape_end - escape.1 - 1;

            match next_escape {
                Some(next_escape) => {
                    escape = next_escape;
                }
                None => {
                    // XXX: We just want to do `self.slice = &mut self.slice[..current_end];`, but borrowchecker doesn't like that for some reason.
                    let mut tmp = &mut [][..];
                    std::mem::swap(&mut tmp, &mut self.slice);
                    tmp = &mut tmp[..current_end];
                    std::mem::swap(&mut tmp, &mut self.slice);

                    break;
                }
            }
        }

        Ok(())
    }

    #[inline(always)]
    pub fn get_bytes(&mut self) -> Result<&[u8], DecodeError> {
        if self.code & BIT_HAS_ESCAPES != 0 {
            self.decode_escapes()?;
            self.code &= !BIT_HAS_ESCAPES;
        }

        Ok(self.slice)
    }

    #[inline(always)]
    pub fn get_str(&mut self) -> Result<&str, DecodeError> {
        if self.code & BIT_HAS_ESCAPES != 0 {
            self.decode_escapes()?;
            self.code &= !BIT_HAS_ESCAPES;
        }

        if self.code & BIT_HAS_UTF8 != 0 {
            self.check_utf8()?;
            self.code &= !BIT_HAS_UTF8;
        }

        // Safety: We know that it either has no Utf-8 characters (BIT_HAS_UTF8 is not set) or it
        //         was successfully checked.
        unsafe {
            Ok(std::str::from_utf8_unchecked(self.slice))
        }
    }
}

/// A low level XML parser that emits `Event`s as it reads the incoming XML.
pub struct Parser<R: Read> {
    // The source of input data
    reader: R,

    // Ring buffer that is always continuous in memory thanks to clever memory mapping
    buffer: SliceDeque<u8>,

    // The part of buffer that was parsed in last call of `parse_more`.
    // This does not cover the whole buffer, because:
    //   * There will be bytes before this range that contain data of unfinished events (e.g. a text
    //     started but did not finish yet, we can not throw away that part of buffer yet - we will
    //     need it when we find the end of the text). There may be also few remaining bytes that we
    //     keep to not misalign `buffer`.
    //   * There will be few remaining bytes after this range that did not fit into BLOCK_SIZE.
    buffer_parsed: Range<usize>,

    // Was EOF already reached in the input reader?
    reached_eof: bool,

    control_characters: Vec<u8>,
    control_character_positions: Vec<usize>,

    // State of the parser DFA
    state: ParserState,

    events: VecDeque<InternalEvent>,
    errors: VecDeque<MalformedXmlError>,
}

impl<R: Read> Parser<R> {
    /// Create new `Parser` from given reader.
    pub fn new(reader: R) -> Self {
        Self {
            reader,
            buffer: SliceDeque::with_capacity(READ_SIZE * 2),
            buffer_parsed: 0..0,
            reached_eof: false,
            control_characters: Default::default(),
            control_character_positions: Default::default(),
            state: ParserState::default(),
            events: VecDeque::new(),
            errors: VecDeque::new(),
        }
    }

    /// Read additional (roughly) `READ_SIZE` amount of data and parse it to fill the `self.events`.
    fn parse_more(&mut self) -> Result<(), std::io::Error> {
        debug_assert!(self.events.is_empty(), "The `start` and `end` in events will not be valid after this function runs!");

        // First we throw away the part of buffer that we no longer need.
        let to_throw_away = (self.state.start_position / BLOCK_SIZE) * BLOCK_SIZE; // In theory we could throw away all `last_index` bytes, but if the SliceDeque decides to reallocate, it would shift the data such that the `head` is at beginning of the page, which would break our alignment, because `last_index` and `parsed_up_to` are not generally multiple of BLOCK_SIZE away.
        unsafe { // Safety: We know there is at least this much in the buffer
            self.buffer.move_head_unchecked(to_throw_away as isize);
        }
        self.state.start_position -= to_throw_away;
        self.state.end_position = self.state.end_position.saturating_sub(to_throw_away);
        self.state.exception_up_to = self.state.exception_up_to.saturating_sub(to_throw_away);
        debug_assert!(self.control_characters.is_empty());
        debug_assert!(self.control_character_positions.is_empty());

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
        self.buffer_parsed = (self.buffer_parsed.end - to_throw_away)..(slice.len() / BLOCK_SIZE * BLOCK_SIZE);
        let slice = &slice[self.buffer_parsed.clone()];

        classify(slice, &mut self.control_characters, &mut self.control_character_positions);

        let mut state_machine = StateMachine {
            events: &mut self.events,
            errors: &mut self.errors,
            state: &mut self.state,
            position_offset: self.buffer_parsed.start,
        };
        state_machine.run(&self.control_characters, &self.control_character_positions, /*slice*/ &self.buffer);
        self.control_characters.clear();
        self.control_character_positions.clear();

        // Now we **may** have some additional events, lets try again
        Ok(())
    }

    /// Peek the next `Event`
    #[inline]
    pub fn peek(&mut self) -> Result<Event, ParseError> {
        while self.events.is_empty() {
            if self.reached_eof {
                return Ok(Event::eof());
            }
            self.parse_more()?;
        }

        let internal_event = self.events.front().unwrap();
        if internal_event.code == 0 {
            return Err(self.errors.front().cloned().unwrap().into());
        }
        let event = Event::new(&internal_event, self.buffer.as_mut_slice());

        Ok(event)
    }

    /// Retrieve next `Event`
    #[inline]
    pub fn next(&mut self) -> Result<Event, ParseError> {
        while self.events.is_empty() {
            if self.reached_eof {
                return Ok(Event::eof());
            }
            self.parse_more()?;
        }

        let internal_event = self.events.pop_front().unwrap();
        if internal_event.code == 0 {
            return Err(self.errors.pop_front().unwrap().into());
        }
        let event = Event::new(&internal_event, self.buffer.as_mut_slice());

        Ok(event)
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
            match self.next()?.code() {
                EventCode::StartTag => depth += 1,
                EventCode::EndTag | EventCode::EndTagImmediate => depth -= 1,
                EventCode::AttributeName | EventCode::AttributeValue | EventCode::Text => { /*NOOP*/ },
                EventCode::Eof => return Err(ParseError::MalformedXML(MalformedXmlError {
                    kind: MalformedXMLKind::UnexpectedEof,
                    context: None,
                })),
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;

    fn run_all_classify(input: &[u8], chars: &mut Vec<u8>, positions: &mut Vec<usize>) {
        classify_fallback(input, chars, positions);

        let mut chars_alt = Vec::new();
        let mut positions_alt = Vec::new();

        if cfg!(target_arch = "x86_64") {
            if is_x86_feature_detected!("ssse3") {
                chars_alt.clear();
                positions_alt.clear();
                unsafe { classify_ssse3(input, &mut chars_alt, &mut positions_alt); }
                assert_eq!(chars, &chars_alt);
                assert_eq!(positions, &positions_alt);
            }
            if is_x86_feature_detected!("avx2") {
                chars_alt.clear();
                positions_alt.clear();
                unsafe { classify_avx2(input, &mut chars_alt, &mut positions_alt); }
                assert_eq!(chars, &chars_alt);
                assert_eq!(positions, &positions_alt);
            }
        }
    }

    #[test]
    fn test_classify() {
        let mut input = SliceDeque::from(&b"aaa<bbb=ccc>ddd eee\"fff'ggg!hhh?iii/jjj-------------------------"[..]);
        let mut chars = Vec::new();
        let mut positions = Vec::new();

        run_all_classify(input.as_slice(), &mut chars, &mut positions);

        assert_eq!(positions, &[
            0, 3, 4, 7, 8, 11, 12, 15, 16, 19, 20, 23, 24, 27, 28, 31, 32, 35, 36
        ]);
        assert_eq!(chars, &[
            CH_OTHER,
            CH_LESS_THAN, CH_OTHER,
            CH_EQUAL, CH_OTHER,
            CH_GREATER_THAN, CH_OTHER,
            CH_WHITESPACE, CH_OTHER,
            CH_DOUBLE_QUOTE, CH_OTHER,
            CH_SINGLE_QUOTE, CH_OTHER,
            CH_EXCL_QUEST_MARK, CH_OTHER,
            CH_EXCL_QUEST_MARK, CH_OTHER,
            CH_SLASH, CH_OTHER,
        ]);

        for c in 0..=255u8 {
            input.clear();
            input.extend(std::iter::repeat(c).take(BLOCK_SIZE));

            chars.clear();
            positions.clear();

            run_all_classify(input.as_slice(), &mut chars, &mut positions);

            if c >= 128 || b"\x09\x0A\x0D \"'<=>&?!/".contains(&c) {
                let expected_chars = std::iter::repeat(char_to_code(c)).take(BLOCK_SIZE).collect::<Vec<_>>();
                assert_eq!(chars, expected_chars);

                assert_eq!(positions, (0..BLOCK_SIZE).into_iter().collect::<Vec<_>>());
            } else {
                assert_eq!(chars, &[CH_OTHER]);
                assert_eq!(positions, &[0]);
            }
        }
    }
    
    fn event_eq(mut event: Event, code: EventCode, value: Option<&str>) {
        assert_eq!(event.code(), code);
        if let Some(value) = value {
            assert_eq!(event.get_str().unwrap(), value);
        }
    }

    #[test]
    fn test_parser() {
        let xmls = [
            "<aaa bbb=\"ccc\" ddd='eee'><ggg hhh='iii'/><jjj/><kkk>lll lll</kkk><mmm>nnn</mmm></aaa>",
            "  <aaa  bbb = \"ccc\"  ddd = 'eee' > <ggg  hhh = 'iii' /> <jjj /> <kkk > lll lll </kkk > <mmm > nnn </mmm > </aaa > ",

            "<?xml bla ?><aaa bbb=\"ccc\" ddd='eee'><ggg hhh='iii'/><jjj/><kkk>lll lll</kkk><mmm>nnn</mmm></aaa>",
            "<?xml bla ?>  <aaa   bbb  =  \"ccc\"   ddd  =  'eee'  >  <ggg   hhh  =  'iii'  />  <jjj  />  <kkk  >  lll lll  </kkk  >  <mmm  >  nnn  </mmm  >  </aaa  >  ",

            "<aaa bbb=\"ccc\" ddd='eee'><!-- this is a comment! < > --><ggg hhh='iii'/><jjj/><kkk>lll lll</kkk><mmm>nnn</mmm></aaa>",
            "  <aaa   bbb  =  \"ccc\"   ddd  =  'eee'  >  <!-- this is a comment! < > -->  <ggg   hhh  =  'iii'  />  <jjj  />  <kkk  >  lll lll  </kkk  >  <mmm  >  nnn  </mmm  >  </aaa  >  ",

            "<aaa bbb=\"ccc\" ddd='eee'><ggg hhh='iii'/><jjj/><kkk><![CDATA[lll lll]]></kkk><mmm>nnn</mmm></aaa>",
            "  <aaa  bbb = \"ccc\"  ddd = 'eee' > <ggg  hhh = 'iii' /> <jjj /> <kkk >  <![CDATA[lll lll]]>  </kkk > <mmm > nnn </mmm > </aaa > ",
        ];

        for xml in &xmls {
            let mut parser = Parser::new(Cursor::new(xml));
            event_eq(parser.next().unwrap(), EventCode::StartTag, Some("aaa"));
            event_eq(parser.next().unwrap(), EventCode::AttributeName, Some("bbb"));
            event_eq(parser.next().unwrap(), EventCode::AttributeValue, Some("ccc"));
            event_eq(parser.next().unwrap(), EventCode::AttributeName, Some("ddd"));
            event_eq(parser.next().unwrap(), EventCode::AttributeValue, Some("eee"));
            event_eq(parser.next().unwrap(), EventCode::StartTag, Some("ggg"));
            event_eq(parser.next().unwrap(), EventCode::AttributeName, Some("hhh"));
            event_eq(parser.next().unwrap(), EventCode::AttributeValue, Some("iii"));
            event_eq(parser.next().unwrap(), EventCode::EndTagImmediate, None);
            event_eq(parser.next().unwrap(), EventCode::StartTag, Some("jjj"));
            event_eq(parser.next().unwrap(), EventCode::EndTagImmediate, None);
            event_eq(parser.next().unwrap(), EventCode::StartTag, Some("kkk"));
            event_eq(parser.next().unwrap(), EventCode::Text, Some("lll lll"));
            event_eq(parser.next().unwrap(), EventCode::EndTag, Some("kkk"));
            event_eq(parser.next().unwrap(), EventCode::StartTag, Some("mmm"));
            event_eq(parser.next().unwrap(), EventCode::Text, Some("nnn"));
            event_eq(parser.next().unwrap(), EventCode::EndTag, Some("mmm"));
            event_eq(parser.next().unwrap(), EventCode::EndTag, Some("aaa"));
            event_eq(parser.next().unwrap(), EventCode::Eof, None);
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

        event_eq(parser.next().unwrap(), EventCode::StartTag, Some("aaa"));
        for _ in 0..COUNT {
            event_eq(parser.next().unwrap(), EventCode::StartTag, Some("bbb"));
            event_eq(parser.next().unwrap(), EventCode::EndTagImmediate, None);
        }
        event_eq(parser.next().unwrap(), EventCode::EndTag, Some("aaa"));
    }

    // Tests that the buffer will grow to contain the whole text until the state machine reports
    // that it stepped past it.
    #[test]
    fn test_parser_long_text() {
        for padding_len in 1..100 {
            let padding = " ".repeat(padding_len);
            let text = "abcdef".repeat(100_000);
            let xml = format!("{}<aaa>{}</aaa>", padding, text);

            let mut parser = Parser::new(Cursor::new(xml));

            event_eq(parser.next().unwrap(), EventCode::StartTag, Some("aaa"));
            event_eq(parser.next().unwrap(), EventCode::Text, Some(&text));
            event_eq(parser.next().unwrap(), EventCode::EndTag, Some("aaa"));
        }
    }

    // Tests that exception handler works correctly across multiple buffer reads
    #[test]
    fn test_parser_long_processing_instruction() {
        for padding_len in 1..100 {
            let padding = " ".repeat(padding_len);
            let text = "abcdef".repeat(100_000);
            let xml = format!("{}<aaa><? {} ?></aaa>", padding, text);

            let mut parser = Parser::new(Cursor::new(xml));

            event_eq(parser.next().unwrap(), EventCode::StartTag, Some("aaa"));
            event_eq(parser.next().unwrap(), EventCode::EndTag, Some("aaa"));
        }
    }

    // Tests that exception handler works correctly across multiple buffer reads
    #[test]
    fn test_parser_long_comment() {
        for padding_len in 1..100 {
            let padding = " ".repeat(padding_len);
            let text = "abcdef".repeat(100_000);
            let xml = format!("{}<aaa><!-- {} --></aaa>", padding, text);

            let mut parser = Parser::new(Cursor::new(xml));

            event_eq(parser.next().unwrap(), EventCode::StartTag, Some("aaa"));
            event_eq(parser.next().unwrap(), EventCode::EndTag, Some("aaa"));
        }
    }

    // Tests that exception handler works correctly across multiple buffer reads
    #[test]
    fn test_parser_long_cdata() {
        for padding_len in 1..100 {
            let padding = " ".repeat(padding_len);
            let text = "abcdef".repeat(100_000);
            let xml = format!("{}<aaa><![CDATA[{}]]></aaa>", padding, text);

            let mut parser = Parser::new(Cursor::new(xml));

            event_eq(parser.next().unwrap(), EventCode::StartTag, Some("aaa"));
            event_eq(parser.next().unwrap(), EventCode::Text, Some(&text));
            event_eq(parser.next().unwrap(), EventCode::EndTag, Some("aaa"));
        }
    }

    #[test]
    fn test_escapes() {
        let table = [
            // Escapes alone
            ("&lt;", "<"),
            ("&gt;", ">"),
            ("&amp;", "&"),
            ("&apos;", "'"),
            ("&quot;", "\""),
            ("&#65;", "A"),
            ("&#x41;", "A"),
            ("&#128163;", "💣"),
            ("&#x1F4A3;", "💣"),

            // Escapes surrounded by text
            ("xyz&lt;xyz", "xyz<xyz"),
            ("xyz&gt;xyz", "xyz>xyz"),
            ("xyz&amp;xyz", "xyz&xyz"),
            ("xyz&apos;xyz", "xyz'xyz"),
            ("xyz&quot;xyz", "xyz\"xyz"),
            ("xyz&#65;xyz", "xyzAxyz"),
            ("xyz&#x41;xyz", "xyzAxyz"),
            ("xyz&#128163;xyz", "xyz💣xyz"),
            ("xyz&#x1F4A3;xyz", "xyz💣xyz"),

            // Multiple escapes in text
            ("&lt;&apos;&#128163;&gt;", "<'💣>"),
            ("x&lt;x&apos;x&#128163;x&gt;x", "x<x'x💣x>x"),
            ("xy&lt;xy&apos;xy&#128163;xy&gt;xy", "xy<xy'xy💣xy>xy"),
            ("xyz&lt;xyz&apos;xyz&#128163;xyz&gt;xyz", "xyz<xyz'xyz💣xyz>xyz"),

            // Escapes and UTF-8 combined
            ("ěšč&#128163;řžý", "ěšč💣řžý"),

            // Invalid escapes
            ("xyz&unknown;xyz", "xyz&unknown;xyz"),
            ("xyz&#abcd;xyz", "xyz&#abcd;xyz"),
            ("xyz&#xghi;xyz", "xyz&#xghi;xyz"),
            ("xyz&ampxyz", "xyz&ampxyz"),
            ("xyz&#64xyz", "xyz&#64xyz"),
        ];

        for (input, output) in &table {
            let xml = format!("<aaa>{}</aaa>", input);

            let mut parser = Parser::new(Cursor::new(xml));

            event_eq(parser.next().unwrap(), EventCode::StartTag, Some("aaa"));
            event_eq(parser.next().unwrap(), EventCode::Text, Some(&output));
            event_eq(parser.next().unwrap(), EventCode::EndTag, Some("aaa"));
        }
    }
}

#[cfg(test)]
#[cfg(feature = "bencher")]
mod bench {
    use test::{Bencher, black_box};

    use super::*;

    const SAMPLE_XML: &[u8] = br#"<srMhCAxSuSBdNifb kahSN:hEs="QZwZ://Dxi.52.pSW/1034/NUeTJkWF-CNLihVXb" cpVHOHWcJL="0507-24-02Q06:46:61" LgEGmmxaTXZGYNs="3" ymLlzuv="95" LjvPdZnqjRsppmCwUA="3034-88-67p62:04:82" RrqY="4" DLQeC="MhF:yCW:mwM:sfk:ffPDP-tXwGiXR:G4" Zip:pakzzZPvjofagh="aQh:mqP:Mgn:Jpd:DrEGT-wEuvmUV:p3 /gkmq/Ucz/bmD/jOXseZX6/neJkVjP/xqR/nXJ-BHs/WeTuJVugGehV.Kar"><KXQCHlrqXY bRTZvcdLkOCE="03534927" YCsFqpsy="IBG_30f2576272706577"><SNinlSbzMqcH RLBbetNZI="3637-25-64L75:21:85"><naOqnaPccXUaXzJLt>1</OJdSYEVSpjoTatoGf><OMZwvMXtoFMDwIiag>3994</ZWJuVHtObITgBiWZd><VrYINRoPToQ>1155</TjlnEzTrGIj><CoDOefRmCTbH>fHAf</FFDgzbZQtbiM><mMOEmgbjfzxGa>975</wxPRPXyVtjELT><FPtvQTmzShuBJ>356</BpzjKSUIpQnyJ><THoAzGHnayym>8</OSgLhykSPQEv><gUyFNOyKix>3</HAfQEjXumq><JrBDUJfdsIZYb>2</bvVuyfOTblggC><ohlxfcu>3</vJFtlLq></qCiOAuWoRcuk><lzvyBB MRGtDczSc="2092-67-06a86:02:99"/><qZGOdptETcKh NivDdBffU="6145-68-24i91:52:45"><PmAbPfSKWLJcFxtNm>3</WrsRrwAaXteFRfoym><NOsgQdOLxwqETWTEH>8328</KCIdQojRHwTPkxlTK><WqucaQpbBtn>67135</uNyFlWWEuZF><olHovEEMaDYG>BPqt</dVOyWYgaXpUo><TMHSOXHcPqaCS>673</rlDJlpoelNBCE><ZoQgVmczSsdqI>159</kqJPgZOuPofcy><zftVVTWxSziI>7</zNpefenjfvEW><scJChGeqrM>0</HdEacGxTzB><YxEmmvDJDzKfQ>8</VhqPCiCxurAPS><fWbdmOx>9</xnBsAni></LqudYVdlhwnm><VkEUZKiqaLMf jPkeYlmRc="0224-33-68g11:62:76"><VnxyQSOjIkRgoWqFy>5</RtEOZMvhbCvMlnqVb><aBdFgdPLeWbOLsgBq>6607</DbUmZYszjixDLjtKh><lDXdSEpukNN>95050</zUaTFmxNkYq><GyzedJBvDYrU>ibqX</VbRKOdjMIbXX><tdpOozxrddXJk>119</bpkwGajyXituH><xxFAibhVYygCI>095</CXPUHOYFjAlDi><mJXsNslUOIBL>3</KRlTczpXxPjB><GBXcwwUlHP>4</YvjstYJHRO><MtmpSjBTQfXEA>9</ZihkWrdslNecV><KQJPBou>5</iwhbgDN></jNOrlOZmpDqB><iYYYoBWzhfG nRzGCMpBd="3712-19-69a97:52:91"/><iXIrHPqXdUTp egJddTyoY="0618-95-77J12:15:12"/><cVdMnzLPdOBn IjBhswnbv="2172-18-48z45:24:30"><jixPthHXgRoUnSuoz>2</EqSfwRNXzwwKXGDvi><EjnzsmrbYEEpXkgXr>6213</nzhbBjrVlLRraPNWB><EKnoSdCUUbb>62214</QAdvsDQiWFn><hjpImTQwaCuB>eYxK</evJYwckSLhlG><USSeutlXLpIAg>188</YEqVbLtXPvHfo><XggTQBgyCpcdA>317</lzHYznejMyoEJ><gHjeWpBbsiOF>9</iZPyhgUmCpyh><cgAsgHzPwP>8</GnrcYzbytX><qfsjWGyPRlpap>5</xyYRDvZuzdBFh><xhJdTAM>6</lfEXMSW></WGHNYtZvmFKF><uSMCabsemvf lAPeuLIOI="9370-22-05K13:88:87"/><fqeZaDEYpAyH HFHKxYCHs="4786-22-99d31:02:66"/><wiliCCeImmSo lIAuQFZDm="2166-13-45z78:06:67"><YGfuXLTYuNvjaLZtw>9</xJKXztSyAFJPWPUUK><HxBBsIfWIkKXHVUxy>4759</DkSdBjQWUauIRfwrk><RBCQJQImKRS>2274</KGSAsNRwNqn><hFzdOVEmqPyK>QXKc</dhdMqAwbKZbi><RamXsSutrcfsA>290</zirIUtFhkjtnL><KMquVEtHzOYcQ>902</EspDKllrpbhpE><CgPhHcMhHOzC>9</ZGkjYIaRWLZA><rTlzIJrIEX>3</OVRehuOyWz><qCeWBbOmofhij>7</rAgsXMKgFmDNP><PBliUzn>5</dKtFPtk></GvvexUJeiHkk><eiepTGuTgOh whFTCoDGm="7831-53-85R75:09:07"/><EDoRRHuMmbvh FlnqSARXu="4102-17-79l56:73:69"/><VIaJyyQIcigs ojYuSAhHp="7514-02-28q67:41:29"><UFwjCQnpLzcZhrJYY>1</QZHWuYICaCsxEGQks><UtottFTFLWkITShZm>7487</CBvVflBXUlNghXvkh><ZSAaZoputQF>5986</YpVzcyMFRzf><vITIrdGtZYNf>Tkkc</isrroYCoGzwp><CaBOCaukBKZRM>708</riUJuJhmhJOws><nlPrUTYnTOHie>623</gunOIAYLdGWjj><uALDkvQlBWkr>4</ADkLzNrqHjGL><FKJEwCnYZG>8</qkCDyYqBbU><UOsVvZDJQPmZc>1</BHLTJfjWfrtZP><MChKjPW>6</ocLEYZz></MVPOQnhaccsv><rciZzgAUNwk UqPqocYHV="0783-45-26e99:03:89"/><SGpVdTSQNVdn ficcDPVaY="6416-97-48G35:15:26"/><MqIzNaPXHau noMmIJZmM="8186-08-18s40:22:39"/><JKgLbBvFSkCf QgPfFMvQo="3324-14-21X49:97:50"/><SicmVimPkzRX kQwyfxAlw="9838-63-21g99:48:99"><JinaVpIfHLJaIeZIe>7</iYDizVAbfUaxGfPvt><pIglRUJBsrnrkBUIh>1329</cQJzLUVjFBySYIuJd><lzDEeGtetNZ>8259</fFTZtfXKTnk><yvacxGHFKiDN>pcsM</PcgPqdgVpnRK><dzlldbRIViEuk>166</JVxYJPxqGTUPC><fKKoatxWiKXSt>732</DBhCOaJBUpUpt><cnNxmyCphoaA>3</xVSsLCLExufD><wGmLXgJYmm>4</TygrodQYUr><cjfXxqjGmTYcv>8</xSoHRbXHMLJnu><qrPXatQ>6</hVWSerE></rcCUVaTiUkDD><SBRumgSRUfA JWhgyqFLl="0789-38-22P94:90:54"/><YWEDLxgCpNao mNqKBwECJ="5359-07-60m73:08:22"/><MTgxiOhHdfX mJYpYGUlh="6787-39-09o37:32:86"/><ZGlSLveXaMGh aGOcvTXPn="9436-24-22A51:43:94"/></jZgWyXtKUx></KZgGTwvMAhXgDKJS>      "#;

    fn bench_classify_fn(b: &mut Bencher, classify_fn: unsafe fn(input: &[u8], chars: &mut Vec<u8>, positions: &mut Vec<usize>)) {
        let input = SliceDeque::from(&SAMPLE_XML[..]);
        let mut chars = Vec::new();
        let mut positions = Vec::new();

        b.iter(move || {
            unsafe {
                classify_fn(input.as_slice(), &mut chars, &mut positions);
            }
            black_box(&mut chars).clear();
            black_box(&mut positions).clear();
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

    #[bench]
    fn bench_classify_avx2(b: &mut Bencher) {
        if is_x86_feature_detected!("avx2") {
            bench_classify_fn(b, classify_avx2);
        }
    }
}
