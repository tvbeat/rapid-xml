//! XML deserializer focused on speed and working with sequences in XML trees.
//!
//! This library provides 3 ways of reading XML, each building on top of the previous one:
//!
//! * `Parser`: Low-level parser that quickly turns a stream of bytes from IO `Read` into a stream
//!             of  events, such as "start tag", "attribute name", "attribute value", "end tag", ...
//! * `Deserializer`: Consumes events from `Parser` and constructs any type that is deserializable
//!                   by serde.
//! * `TreeDeserializer`: Deserializes sequences of (optionally nested) types from XML trees.

#![warn(missing_docs)]

#![cfg_attr(feature = "bencher", feature(test))]
#[cfg(feature = "bencher")]
extern crate test;

pub use de::Deserializer;
pub use parser::{DeferredString, Error, Event, MalformedXMLKind, Parser};
pub use tree::{ElementDeserialize, ElementEnter, ElementEnterDeserialize, XmlPath, TreeDeserializer};

mod de;
mod tree;
mod parser;
