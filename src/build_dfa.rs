use std::collections::{HashMap, HashSet, BTreeMap};
use std::convert::{TryInto, TryFrom};

use num_enum::TryFromPrimitive;

// Layout of Event:
//   0bUE00_0AAA
//     U          ..  has_utf8
//      E         ..  has_escapes
//           AAA  ..  event code

const BIT_HAS_UTF8: u8 =    0b1000_0000;
const BIT_HAS_ESCAPES: u8 = 0b0100_0000;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
enum Event {
    None = 0, // Not real event. 0 is unused, all zeroes in bottom octet makes the exception State.

    StartTag =        0o1,

    EndTag =          0o2,
    EndTagImmediate = 0o3,

    Text =            0o4,

    AttributeName =   0o5,
    AttributeValue =  0o6,

    Eof =             0o7,
}

// Layout of Character:
//   0bUE00_CCCC
//     U          ..  set utf-8 flag
//      E         ..  set escaped flag
//          CCCC  ..  character code
//
// Layout of State:
//   0b000B_BAAA
//        B_BAAA  ..  state code
//           AAA  ..  event code that would be emitted when transitioning **from** this state
//
// Layout of Transition:
//   0bXYZB_BAAA
//     X          ..  emit flag, unset utf8+escapes
//      Y         ..  save start flag
//       Z        ..  save end flag

const BIT_EMIT: u8 = 0b1000_0000;
const BIT_SAVE_START: u8 = 0b0100_0000;
const BIT_SAVE_END: u8 = 0b0010_0000;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, TryFromPrimitive, PartialOrd, Ord)]
#[repr(u8)]
enum State {
    Exception = 0,

    Outside = Event::Eof as u8, // emits Event::Eof

    TagStart = 0o10,
    TagEnd = 0o11,

    TagName = Event::StartTag as u8, // emits Event::StartTag
    TagEndName = Event::EndTag as u8, // emits Event::EndTag

    InTag = 0o12,
    InTagEnd = 0o13,

    AttrName = Event::AttributeName as u8, // emits Event::AttributeName

    AfterAttrName = 0o15, // Space between attribute name and '=' sign

    AfterAttrEq = 0o17,

    AttrValueDoubleQuotedOpen = 0o20,
    AttrValueDoubleQuoted = Event::AttributeValue as u8, // emits Event::AttributeValue

    AttrValueSingleQuotedOpen = 0o21,
    AttrValueSingleQuoted = Event::AttributeValue as u8 | 0o10, // emits Event::AttributeValue

    AfterAttrValue = 0o22,

    AfterImmediateEndTag = Event::EndTagImmediate as u8, // emits Event::EndTagImmediate

    InText = Event::Text as u8, // emits Event::Text
    InTextEndWhitespace = Event::Text as u8 | 0o10, // emits Event::Text
}

const STATES: [State; 19] = [
    State::Exception,
    State::Outside,
    State::TagStart,
    State::TagEnd,
    State::TagName,
    State::TagEndName,
    State::InTag,
    State::InTagEnd,
    State::AttrName,
    State::AfterAttrName,
    State::AfterAttrEq,
    State::AttrValueDoubleQuotedOpen,
    State::AttrValueDoubleQuoted,
    State::AttrValueSingleQuotedOpen,
    State::AttrValueSingleQuoted,
    State::AfterAttrValue,
    State::AfterImmediateEndTag,
    State::InText,
    State::InTextEndWhitespace,
];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct Trans {
    new_state: State,
    save_start_position: bool,
    save_end_position: bool,

    emit: Event,
}

impl Trans {
    fn new(new_state: State, save_start_position: bool, save_end_position: bool, emit: Event) -> Self {
        Self {
            new_state,
            save_start_position,
            save_end_position,
            emit,
        }
    }

    fn as_u8(&self) -> u8 {
        let mut n = self.new_state as u8;
        if self.emit != Event::None {
            n |= 0b1000_0000;
        }
        if self.save_start_position {
            n |= 0b0100_0000;
        }
        if self.save_end_position {
            n |= 0b0010_0000;
        }
        n
    }
}

macro_rules! add {
    ($table:expr, ($state:expr, [$($ch:expr),+]) => $new_state:expr, $save_start_position:expr, $save_end_position:expr) => {
        $(
            let o = $table.insert(($state, $ch), Trans::new($new_state, $save_start_position, $save_end_position, Event::None));
            assert!(o.is_none(), "Duplicated transition!");
        )+
    };

    ($table:expr, ($state:expr, [$($ch:expr),+]) => $new_state:expr, $save_start_position:expr, $save_end_position:expr, $emit:expr) => {
        $(
            let o = $table.insert(($state, $ch), Trans::new($new_state, $save_start_position, $save_end_position, $emit));
            assert!(o.is_none(), "Duplicated transition!");
        )+
    };

    ($table:expr, ($state:expr, $ch:expr) => $new_state:expr, $save_start_position:expr, $save_end_position:expr) => {
        add!($table, ($state, $ch) => $new_state, $save_start_position, $save_end_position, Event::None);
    };

    ($table:expr, ($state:expr, $ch:expr) => $new_state:expr, $save_start_position:expr, $save_end_position:expr, $emit:expr) => {
        let o = $table.insert(($state, $ch), Trans::new($new_state, $save_start_position, $save_end_position, $emit));
        assert!(o.is_none(), "Duplicated transition!");
    };
}

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

const ALPHABET: [u8; 9] = [
    CH_OTHER,
    // CH_UTF8,      // not part of alphabet
    // CH_AMPERSAND, // not part of alphabet
    CH_DOUBLE_QUOTE,
    CH_SINGLE_QUOTE,
    CH_WHITESPACE,
    CH_EXCL_QUEST_MARK,
    CH_SLASH,
    CH_LESS_THAN,
    CH_EQUAL,
    CH_GREATER_THAN,
];

type Table = HashMap<(State, u8), Trans>;

fn build_state_machine() -> Table {
    use State::*;

    let mut table = HashMap::new();

    // Beginning of a tag
    add!(table, (Outside, CH_LESS_THAN) => TagStart, false, false);

    // Tag start
    add!(table, (TagStart, CH_OTHER) => TagName, true, false);

    add!(table, (TagName, CH_OTHER) => TagName, false, false);

    add!(table, (TagName, CH_WHITESPACE)   => InTag, false, true, Event::StartTag);
    add!(table, (TagName, CH_GREATER_THAN) => Outside, false, true, Event::StartTag);
    add!(table, (TagName, CH_SLASH) => AfterImmediateEndTag, false, true, Event::StartTag);

    add!(table, (InTag, CH_WHITESPACE) => InTag, false, false);

    // Attributes
    add!(table, (InTag, CH_OTHER) => AttrName, true, false);

    add!(table, (AttrName, CH_OTHER) => AttrName, false, false);
    add!(table, (AttrName, CH_WHITESPACE) => AfterAttrName, false, true, Event::AttributeName);
    add!(table, (AttrName, CH_EQUAL)      => AfterAttrEq,   false, true, Event::AttributeName);

    add!(table, (AfterAttrName, CH_WHITESPACE) => AfterAttrName, false, false);
    add!(table, (AfterAttrName, CH_EQUAL)      => AfterAttrEq, false, false);

    add!(table, (AfterAttrEq, CH_WHITESPACE)   => AfterAttrEq, false, false);
    add!(table, (AfterAttrEq, CH_DOUBLE_QUOTE) => AttrValueDoubleQuotedOpen, false, false);
    add!(table, (AfterAttrEq, CH_SINGLE_QUOTE) => AttrValueSingleQuotedOpen, false, false);

    add!(table, (AttrValueDoubleQuotedOpen, [CH_SINGLE_QUOTE, CH_WHITESPACE, CH_EQUAL, CH_GREATER_THAN, CH_SLASH, CH_EXCL_QUEST_MARK, CH_OTHER]) => AttrValueDoubleQuoted, true, false);
    add!(table, (AttrValueSingleQuotedOpen, [CH_DOUBLE_QUOTE, CH_WHITESPACE, CH_EQUAL, CH_GREATER_THAN, CH_SLASH, CH_EXCL_QUEST_MARK, CH_OTHER]) => AttrValueSingleQuoted, true, false);

    add!(table, (AttrValueDoubleQuoted, [CH_SINGLE_QUOTE, CH_WHITESPACE, CH_EQUAL, CH_GREATER_THAN, CH_SLASH, CH_EXCL_QUEST_MARK, CH_OTHER]) => AttrValueDoubleQuoted, false, false);
    add!(table, (AttrValueSingleQuoted, [CH_DOUBLE_QUOTE, CH_WHITESPACE, CH_EQUAL, CH_GREATER_THAN, CH_SLASH, CH_EXCL_QUEST_MARK, CH_OTHER]) => AttrValueSingleQuoted, false, false);

    add!(table, (AttrValueDoubleQuoted, CH_DOUBLE_QUOTE) => AfterAttrValue, false, true, Event::AttributeValue);
    add!(table, (AttrValueSingleQuoted, CH_SINGLE_QUOTE) => AfterAttrValue, false, true, Event::AttributeValue);

    add!(table, (AfterAttrValue, CH_WHITESPACE)   => InTag, false, false);
    add!(table, (AfterAttrValue, CH_GREATER_THAN) => Outside, false, false);
    add!(table, (AfterAttrValue, CH_SLASH) => AfterImmediateEndTag, false, false);

    // Tag end immediate
    add!(table, (InTag, CH_GREATER_THAN) => Outside, false, false);
    add!(table, (InTag, CH_SLASH) => AfterImmediateEndTag, false, false);

    add!(table, (AfterImmediateEndTag, CH_GREATER_THAN) => Outside, false, false, Event::EndTagImmediate);

    // Tag end
    add!(table, (TagStart, CH_SLASH) => TagEnd, false, false);

    add!(table, (TagEnd, CH_OTHER) => TagEndName, true, false);

    add!(table, (TagEndName, CH_OTHER) => TagEndName, false, false);
    add!(table, (TagEndName, CH_WHITESPACE) => InTagEnd, false, true, Event::EndTag);

    add!(table, (TagEndName, CH_GREATER_THAN) => Outside, false, true, Event::EndTag);

    add!(table, (InTagEnd, CH_WHITESPACE) => InTagEnd, false, false);
    add!(table, (InTagEnd, CH_GREATER_THAN) => Outside, false, false);

    // Text
    add!(table, (Outside, CH_WHITESPACE) => Outside,       false, false);

    add!(table, (Outside,              [CH_DOUBLE_QUOTE, CH_SINGLE_QUOTE, CH_EQUAL, CH_GREATER_THAN, CH_SLASH, CH_EXCL_QUEST_MARK, CH_OTHER]) => InText, true, false);
    add!(table, (InText,               [CH_DOUBLE_QUOTE, CH_SINGLE_QUOTE, CH_EQUAL, CH_GREATER_THAN, CH_SLASH, CH_EXCL_QUEST_MARK, CH_OTHER]) => InText, false, false);
    add!(table, (InTextEndWhitespace,  [CH_DOUBLE_QUOTE, CH_SINGLE_QUOTE, CH_EQUAL, CH_GREATER_THAN, CH_SLASH, CH_EXCL_QUEST_MARK, CH_OTHER]) => InText, false, false);

    add!(table, (InText,              CH_WHITESPACE) => InTextEndWhitespace, false, true);
    add!(table, (InTextEndWhitespace, CH_WHITESPACE) => InTextEndWhitespace, false, false);

    add!(table, (InText,              CH_LESS_THAN) => TagStart, false, true,  Event::Text);
    add!(table, (InTextEndWhitespace, CH_LESS_THAN) => TagStart, false, false, Event::Text);

    // Processing instruction, Comment, CDATA, etc
    add!(table, (TagStart, CH_EXCL_QUEST_MARK) => Exception, false, false); // This would be exception even without this, but this is here to make it explicit. This will let us handle comments, CDATA, ENTITY, etc.

    table
}

fn print_lut(table: &Table) {
    let mut states = STATES.to_vec();
    states.sort();

    let mut alphabet = ALPHABET.to_vec();
    alphabet.sort();

    print!("    ");
    for c in &alphabet {
        print!("  {:02x}  ", c);
    }
    println!();

    for state in &states {
        print!("/*   {:02x}:  */", *state as u8);
        for c in &alphabet {
            let trans = table.get(&(*state, *c)).map(|trans| trans.as_u8()).unwrap_or(0);
            print!(" 0x{:02x},", trans);
        }
        println!();
    }


    println!();
    println!();


    print!("    ");
    for i in 0..16 {
        print!("{:02x} ", i);
    }
    println!();

    let mut i = 0;
    for state in &states {
        for c in &alphabet {
            if i % 16 == 0 {
                print!("/*   {:02x}:  */", i / 16 * 16);
            }

            let trans = table.get(&(*state, *c)).map(|trans| trans.as_u8()).unwrap_or(0);
            print!(" 0x{:02x},", trans);

            if i % 16 == 15 {
                println!();
            }
            i += 1;
        }
    }
}

fn main() {
    let table = build_state_machine();

    print_lut(&table);
}
