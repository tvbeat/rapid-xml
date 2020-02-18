//! Contains serde Deserializer build on top of `Parser` from `parse`.

use std::borrow::Cow;
use std::convert::TryInto;
use std::fmt::Display;
use std::io::Read;

use inlinable_string::InlinableString;
use serde::{Deserializer as _, forward_to_deserialize_any};
use serde::de::{DeserializeSeed, IntoDeserializer, Visitor};

use crate::parser::{DeferredString, Event, ParseError, Parser};

/// Error while parsing or deserializing
#[derive(Debug)]
pub enum DeserializeError {
    /// Error from underlying parser
    Parsing(ParseError),

    /// Error parsing integer
    ParseInt(btoi::ParseIntegerError),

    /// Error parsing floating point number
    ParseFloat(std::num::ParseFloatError),

    /// Error parsing bool
    ParseBool(std::str::ParseBoolError),

    /// Error deserializing character - there was none or too many characters
    NotOneCharacter,

    /// EOF came too early.
    UnexpectedEof,

    /// Deserializer was expecting an element, but found something else
    ExpectedElement,

    /// Deserializer was expecting text, but found none
    ExpectedText,

    /// Custom error from Serde
    Custom(String),
}

impl Display for DeserializeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        match self {
            DeserializeError::Parsing(err) => Display::fmt(err, f),
            DeserializeError::ParseInt(err) => Display::fmt(err, f),
            DeserializeError::ParseFloat(err) => Display::fmt(err, f),
            DeserializeError::ParseBool(err) => Display::fmt(err, f),
            DeserializeError::NotOneCharacter => write!(f, "Expected character, but found string with more or less than 1 character."),
            DeserializeError::UnexpectedEof => write!(f, "Unexpected EOF."),
            DeserializeError::ExpectedElement => write!(f, "Expected element, but found something else."),
            DeserializeError::ExpectedText => write!(f, "Expected text, but found none."),
            DeserializeError::Custom(string) => write!(f, "{}", string),
        }
    }
}

impl std::error::Error for DeserializeError {}

impl From<ParseError> for DeserializeError {
    fn from(err: ParseError) -> Self {
        DeserializeError::Parsing(err)
    }
}

impl From<btoi::ParseIntegerError> for DeserializeError {
    fn from(err: btoi::ParseIntegerError) -> Self {
        DeserializeError::ParseInt(err)
    }
}

impl From<std::num::ParseFloatError> for DeserializeError {
    fn from(err: std::num::ParseFloatError) -> Self {
        DeserializeError::ParseFloat(err)
    }
}

impl From<std::str::ParseBoolError> for DeserializeError {
    fn from(err: std::str::ParseBoolError) -> Self {
        DeserializeError::ParseBool(err)
    }
}

impl serde::de::Error for DeserializeError {
    fn custom<T: std::fmt::Display>(msg: T) -> Self {
        DeserializeError::Custom(msg.to_string())
    }
}

/// Serde `Deserializer` that builds types from XML [`Event`]s produced by [`Parser`]
pub struct Deserializer<'a, R: Read> {
    parser: &'a mut Parser<R>,
    opening_tag: InlinableString,
    only_attributes: bool,
}

impl<'a, R: Read> Deserializer<'a, R> {
    /// Create new `TreeDeserializer` from given `Parser`.
    pub fn new(parser: &'a mut Parser<R>) -> Result<Self, DeserializeError> {
        // Enter the root element
        loop {
            let event = parser.next()?;
            match event {
                Event::StartTag(tag_name) => {
                    let opening_tag = tag_name.try_into()?;
                    return Ok(Self::new_inside_tag(parser, opening_tag, false));
                }

                Event::Text(_) => {
                    /* Text before the first tag, we ignore it... */
                }

                _ => {
                    return Err(DeserializeError::ExpectedElement);
                }
            }
        }
    }

    // TODO: Should we expose something like this to public?
    pub(crate) fn new_inside_tag(parser: &'a mut Parser<R>, opening_tag: InlinableString, only_attributes: bool) -> Self {
        Self {
            parser,
            opening_tag,
            only_attributes,
        }
    }

    /// Call given callback on the content of next `Text` event
    fn with_next_text<'x: 'a, F: FnOnce(DeferredString) -> Result<O, DeserializeError>, O>(&'x mut self, f: F) -> Result<O, DeserializeError> {
        let mut depth = 1;
        while depth > 0 {
            let event = self.parser.next()?;
            match event {
                Event::StartTag(_) => depth += 1,
                Event::EndTag(_) | Event::EndTagImmediate => depth -= 1,
                Event::AttributeName(_) | Event::AttributeValue(_) | Event::StartTagDone => { /*NOOP*/ },
                Event::Text(value) => {
                    let out = f(value);
                    self.parser.finish_tag(depth)?;
                    return out;
                }
                Event::Eof => break,
            }
        }

        return Err(DeserializeError::ExpectedText);
    }
}

impl<'de: 'a, 'a, R: Read> serde::Deserializer<'de> for &'a mut Deserializer<'a, R> {
    type Error = DeserializeError;

    fn deserialize_any<V: Visitor<'de>>(self, visitor: V) -> Result<<V as Visitor<'de>>::Value, Self::Error> {
        // TODO: Correct option?
        self.deserialize_string(visitor)
    }

    fn deserialize_bool<V: Visitor<'de>>(self, visitor: V) -> Result<<V as Visitor<'de>>::Value, Self::Error> {
        // TODO: Do we want to accept other strings in addition to "true" and "false"?
        self.with_next_text(|t| visitor.visit_bool(t.to_str()?.parse()?))
    }

    fn deserialize_i8<V: Visitor<'de>>(self, visitor: V) -> Result<<V as Visitor<'de>>::Value, Self::Error> {
        self.with_next_text(|t| visitor.visit_i8(btoi::btoi(t.to_bytes().as_ref())?))
    }

    fn deserialize_i16<V: Visitor<'de>>(self, visitor: V) -> Result<<V as Visitor<'de>>::Value, Self::Error> {
        self.with_next_text(|t| visitor.visit_i16(btoi::btoi(t.to_bytes().as_ref())?))
    }

    fn deserialize_i32<V: Visitor<'de>>(self, visitor: V) -> Result<<V as Visitor<'de>>::Value, Self::Error> {
        self.with_next_text(|t| visitor.visit_i32(btoi::btoi(t.to_bytes().as_ref())?))
    }

    fn deserialize_i64<V: Visitor<'de>>(self, visitor: V) -> Result<<V as Visitor<'de>>::Value, Self::Error> {
        self.with_next_text(|t| visitor.visit_i64(btoi::btoi(t.to_bytes().as_ref())?))
    }

    fn deserialize_u8<V: Visitor<'de>>(self, visitor: V) -> Result<<V as Visitor<'de>>::Value, Self::Error> {
        self.with_next_text(|t| visitor.visit_u8(btoi::btou(t.to_bytes().as_ref())?))
    }

    fn deserialize_u16<V: Visitor<'de>>(self, visitor: V) -> Result<<V as Visitor<'de>>::Value, Self::Error> {
        self.with_next_text(|t| visitor.visit_u16(btoi::btou(t.to_bytes().as_ref())?))
    }

    fn deserialize_u32<V: Visitor<'de>>(self, visitor: V) -> Result<<V as Visitor<'de>>::Value, Self::Error> {
        self.with_next_text(|t| visitor.visit_u32(btoi::btou(t.to_bytes().as_ref())?))
    }

    fn deserialize_u64<V: Visitor<'de>>(self, visitor: V) -> Result<<V as Visitor<'de>>::Value, Self::Error> {
        self.with_next_text(|t| visitor.visit_u64(btoi::btou(t.to_bytes().as_ref())?))
    }

    fn deserialize_f32<V: Visitor<'de>>(self, visitor: V) -> Result<<V as Visitor<'de>>::Value, Self::Error> {
        self.with_next_text(|t| visitor.visit_f32(t.to_str()?.parse()?))
    }

    fn deserialize_f64<V: Visitor<'de>>(self, visitor: V) -> Result<<V as Visitor<'de>>::Value, Self::Error> {
        self.with_next_text(|t| visitor.visit_f64(t.to_str()?.parse()?))
    }

    fn deserialize_char<V: Visitor<'de>>(self, visitor: V) -> Result<<V as Visitor<'de>>::Value, Self::Error> {
        self.with_next_text(|string| {
            let string = string.to_str()?;
            if string.len() == 1 {
                visitor.visit_char(string.chars().next().unwrap()) // NOTE(unwrap): We know there is exactly one character.
            } else {
                return Err(DeserializeError::NotOneCharacter);
            }
        })
    }

    fn deserialize_str<V: Visitor<'de>>(self, visitor: V) -> Result<<V as Visitor<'de>>::Value, Self::Error> {
        self.deserialize_string(visitor)
    }

    fn deserialize_string<V: Visitor<'de>>(self, visitor: V) -> Result<<V as Visitor<'de>>::Value, Self::Error> {
        self.with_next_text(|text| {
            match text.to_str()? {
                Cow::Borrowed(str) => visitor.visit_str(str),
                Cow::Owned(string) => visitor.visit_string(string),
            }
        })
    }

    fn deserialize_bytes<V: Visitor<'de>>(self, _visitor: V) -> Result<<V as Visitor<'de>>::Value, Self::Error> {
        unimplemented!()
    }

    fn deserialize_byte_buf<V: Visitor<'de>>(self, _visitor: V) -> Result<<V as Visitor<'de>>::Value, Self::Error> {
        unimplemented!()
    }

    fn deserialize_option<V: Visitor<'de>>(self, visitor: V) -> Result<<V as Visitor<'de>>::Value, Self::Error> {
        // We don't really have a "null" value. The only `Option`s that will work are for map keys that are
        // not present. If someone is trying to deserialize `Option` in other place, then we will always give
        // `Some` and try to deserialize the content.
        visitor.visit_some(self)
    }

    fn deserialize_unit<V: Visitor<'de>>(self, _visitor: V) -> Result<<V as Visitor<'de>>::Value, Self::Error> {
        unimplemented!()
    }

    fn deserialize_unit_struct<V: Visitor<'de>>(self, _name: &'static str, _visitor: V) -> Result<<V as Visitor<'de>>::Value, Self::Error> {
        unimplemented!()
    }

    fn deserialize_newtype_struct<V: Visitor<'de>>(self, _name: &'static str, _visitor: V) -> Result<<V as Visitor<'de>>::Value, Self::Error> {
        unimplemented!()
    }

    fn deserialize_seq<V: Visitor<'de>>(self, _visitor: V) -> Result<<V as Visitor<'de>>::Value, Self::Error> {
        unimplemented!()
    }

    fn deserialize_tuple<V: Visitor<'de>>(self, _len: usize, _visitor: V) -> Result<<V as Visitor<'de>>::Value, Self::Error> {
        unimplemented!()
    }

    fn deserialize_tuple_struct<V: Visitor<'de>>(self, _name: &'static str, _len: usize, _visitor: V) -> Result<<V as Visitor<'de>>::Value, Self::Error> {
        unimplemented!()
    }

    fn deserialize_map<V: Visitor<'de>>(self, visitor: V) -> Result<<V as Visitor<'de>>::Value, Self::Error> {
        struct MapAccess<'a, R: Read> {
            parser: &'a mut Parser<R>,
            in_tag: Option<InlinableString>,
            only_attributes: bool,
        }

        impl<'a, 'de: 'a, R: Read> serde::de::MapAccess<'de> for MapAccess<'a, R> {
            type Error = DeserializeError;

            fn next_key_seed<K: DeserializeSeed<'de>>(&mut self, seed: K) -> Result<Option<<K as DeserializeSeed<'de>>::Value>, Self::Error> {
                loop {
                    return match self.parser.next()? {
                        Event::AttributeName(name) =>
                            seed.deserialize(name.to_str()?.into_deserializer()).map(Some),
                        Event::StartTag(name) => {
                            let tag_name: InlinableString = name.try_into()?;
                            let out = seed.deserialize(tag_name.into_deserializer()).map(Some);
                            self.in_tag = Some(tag_name);
                            out
                        }
                        Event::EndTag(_) | Event::EndTagImmediate => Ok(None),
                        Event::StartTagDone => {
                            if self.only_attributes {
                                Ok(None)
                            } else {
                                continue;
                            }
                        }
                        e => unreachable!("Parser should have never given us {:?} event in this place.", e),
                    }
                }
            }

            fn next_value_seed<V: DeserializeSeed<'de>>(&mut self, seed: V) -> Result<<V as DeserializeSeed<'de>>::Value, Self::Error> {
                if let Some(tag_name) = self.in_tag.take() {
                    return seed.deserialize(&mut Deserializer {
                        parser: self.parser,
                        opening_tag: tag_name,
                        only_attributes: self.only_attributes, // Should be true at this point.
                    });
                }

                match self.parser.next()? {
                    Event::AttributeValue(string) =>
                        seed.deserialize(ParseDeserializer {
                            string,
                        }),
                    e => unreachable!("Parser should have never given us {:?} event in this place.", e),
                }
            }
        }

        visitor.visit_map(MapAccess {
            parser: &mut *self.parser,
            in_tag: None,
            only_attributes: self.only_attributes,
        })
    }

    fn deserialize_struct<V: Visitor<'de>>(self, _name: &'static str, _fields: &'static [&'static str], visitor: V) -> Result<<V as Visitor<'de>>::Value, Self::Error> {
        self.deserialize_map(visitor)
    }

    fn deserialize_enum<V: Visitor<'de>>(self, _name: &'static str, _variants: &'static [&'static str], visitor: V) -> Result<<V as Visitor<'de>>::Value, Self::Error> {
        //panic!("{:?}", self.parser.next());

        struct EnumAccess<'a, R: Read> {
            parser: &'a mut Parser<R>,
            opening_tag: InlinableString,
        }

        impl<'a, 'de: 'a, R: Read> serde::de::EnumAccess<'de> for EnumAccess<'a, R> {
            type Error = DeserializeError;
            type Variant = Self;

            fn variant_seed<V: DeserializeSeed<'de>>(self, seed: V) -> Result<(<V as DeserializeSeed<'de>>::Value, Self::Variant), Self::Error> {
                seed.deserialize(&mut Deserializer {
                    parser: self.parser,
                    opening_tag: self.opening_tag.clone(),
                    only_attributes: false,
                }).map(|value| (value, self))
            }
        }

        impl<'a, 'de: 'a, R: Read> serde::de::VariantAccess<'de> for EnumAccess<'a, R> {
            type Error = DeserializeError;

            fn unit_variant(self) -> Result<(), Self::Error> {
                self.parser.finish_tag(1)?; // ?

                Ok(())
            }

            fn newtype_variant_seed<T: DeserializeSeed<'de>>(self, seed: T) -> Result<<T as DeserializeSeed<'de>>::Value, Self::Error> {
                let deserializer = &mut Deserializer {
                    parser: self.parser,
                    opening_tag: self.opening_tag,
                    only_attributes: false,
                };
                seed.deserialize(deserializer)
            }

            fn tuple_variant<V: Visitor<'de>>(self, _len: usize, visitor: V) -> Result<<V as Visitor<'de>>::Value, Self::Error> {
                let deserializer = &mut Deserializer {
                    parser: self.parser,
                    opening_tag: self.opening_tag,
                    only_attributes: false,
                };
                deserializer.deserialize_seq(visitor)
            }

            fn struct_variant<V: Visitor<'de>>(self, fields: &'static [&'static str], visitor: V) -> Result<<V as Visitor<'de>>::Value, Self::Error> {
                let deserializer = &mut Deserializer {
                    parser: self.parser,
                    opening_tag: self.opening_tag,
                    only_attributes: false,
                };
                deserializer.deserialize_struct("", fields, visitor)
            }
        }

        visitor.visit_enum(EnumAccess {
            parser: self.parser,
            opening_tag: self.opening_tag.clone(),
        })
    }

    fn deserialize_identifier<V: Visitor<'de>>(self, visitor: V) -> Result<<V as Visitor<'de>>::Value, Self::Error> {
        visitor.visit_str(&self.opening_tag)
    }

    fn deserialize_ignored_any<V: Visitor<'de>>(self, visitor: V) -> Result<<V as Visitor<'de>>::Value, Self::Error> {
        self.parser.finish_tag(1)?;

        visitor.visit_unit() // ?
    }
}

struct ParseDeserializer<'a> {
    string: DeferredString<'a>,
}

impl<'de: 'a, 'a> serde::Deserializer<'de> for ParseDeserializer<'a> {
    type Error = DeserializeError;

    fn deserialize_any<V: Visitor<'de>>(self, visitor: V) -> Result<<V as Visitor<'de>>::Value, Self::Error> {
        match self.string.to_str()? {
            Cow::Owned(string) => visitor.visit_string(string),
            Cow::Borrowed(str) => visitor.visit_str(str),
        }
    }

    fn deserialize_bool<V: Visitor<'de>>(self, visitor: V) -> Result<<V as Visitor<'de>>::Value, Self::Error> {
        // TODO: Do we want to accept other strings in addition to "true" and "false"?
        visitor.visit_bool(self.string.to_str()?.parse()?)
    }

    fn deserialize_i8<V: Visitor<'de>>(self, visitor: V) -> Result<<V as Visitor<'de>>::Value, Self::Error> {
        visitor.visit_i8(btoi::btoi(self.string.to_bytes().as_ref())?)
    }

    fn deserialize_i16<V: Visitor<'de>>(self, visitor: V) -> Result<<V as Visitor<'de>>::Value, Self::Error> {
        visitor.visit_i16(btoi::btoi(self.string.to_bytes().as_ref())?)
    }

    fn deserialize_i32<V: Visitor<'de>>(self, visitor: V) -> Result<<V as Visitor<'de>>::Value, Self::Error> {
        visitor.visit_i32(btoi::btoi(self.string.to_bytes().as_ref())?)
    }

    fn deserialize_i64<V: Visitor<'de>>(self, visitor: V) -> Result<<V as Visitor<'de>>::Value, Self::Error> {
        visitor.visit_i64(btoi::btoi(self.string.to_bytes().as_ref())?)
    }

    fn deserialize_u8<V: Visitor<'de>>(self, visitor: V) -> Result<<V as Visitor<'de>>::Value, Self::Error> {
        visitor.visit_u8(btoi::btou(self.string.to_bytes().as_ref())?)
    }

    fn deserialize_u16<V: Visitor<'de>>(self, visitor: V) -> Result<<V as Visitor<'de>>::Value, Self::Error> {
        visitor.visit_u16(btoi::btou(self.string.to_bytes().as_ref())?)
    }

    fn deserialize_u32<V: Visitor<'de>>(self, visitor: V) -> Result<<V as Visitor<'de>>::Value, Self::Error> {
        visitor.visit_u32(btoi::btou(self.string.to_bytes().as_ref())?)
    }

    fn deserialize_u64<V: Visitor<'de>>(self, visitor: V) -> Result<<V as Visitor<'de>>::Value, Self::Error> {
        visitor.visit_u64(btoi::btou(self.string.to_bytes().as_ref())?)
    }

    fn deserialize_f32<V: Visitor<'de>>(self, visitor: V) -> Result<<V as Visitor<'de>>::Value, Self::Error> {
        visitor.visit_f32(self.string.to_str()?.parse()?)
    }

    fn deserialize_f64<V: Visitor<'de>>(self, visitor: V) -> Result<<V as Visitor<'de>>::Value, Self::Error> {
        visitor.visit_f64(self.string.to_str()?.parse()?)
    }

    fn deserialize_char<V: Visitor<'de>>(self, visitor: V) -> Result<<V as Visitor<'de>>::Value, Self::Error> {
        let s = self.string.to_str()?;
        if s.len() == 1 {
            visitor.visit_char(s.chars().next().unwrap()) // NOTE(unwrap): We know there is exactly one character.
        } else {
            Err(DeserializeError::NotOneCharacter)
        }
    }

    fn deserialize_str<V: Visitor<'de>>(self, visitor: V) -> Result<<V as Visitor<'de>>::Value, Self::Error> {
        visitor.visit_str(self.string.to_str()?.as_ref())
    }

    fn deserialize_string<V: Visitor<'de>>(self, visitor: V) -> Result<<V as Visitor<'de>>::Value, Self::Error> {
        visitor.visit_string(self.string.to_str()?.into_owned())
    }

    fn deserialize_bytes<V: Visitor<'de>>(self, visitor: V) -> Result<<V as Visitor<'de>>::Value, Self::Error> {
        match self.string.to_bytes() {
            Cow::Owned(vec) => visitor.visit_byte_buf(vec),
            Cow::Borrowed(bytes) => visitor.visit_bytes(bytes),
        }
    }

    fn deserialize_byte_buf<V: Visitor<'de>>(self, visitor: V) -> Result<<V as Visitor<'de>>::Value, Self::Error> {
        self.deserialize_bytes(visitor) // TODO: Correct?
    }

    fn deserialize_option<V: Visitor<'de>>(self, visitor: V) -> Result<<V as Visitor<'de>>::Value, Self::Error> {
        // We don't really have a "null" value. The only `Option`s that will work are for map keys that are
        // not present. If someone is trying to deserialize `Option` in other place, then we will always give
        // `Some` and try to deserialize the content.
        visitor.visit_some(self)
    }

    forward_to_deserialize_any! { unit unit_struct newtype_struct seq tuple tuple_struct map struct enum }

    fn deserialize_identifier<V: Visitor<'de>>(self, _visitor: V) -> Result<<V as Visitor<'de>>::Value, Self::Error> {
        unimplemented!()
    }

    fn deserialize_ignored_any<V: Visitor<'de>>(self, visitor: V) -> Result<<V as Visitor<'de>>::Value, Self::Error> {
        self.deserialize_any(visitor) // TODO: Proper way?
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use serde::Deserialize as _;
    use serde_derive::Deserialize;

    use super::*;

    #[test]
    fn test_deserializer() {
        #[derive(Clone, Debug, Deserialize, PartialEq)]
        struct MyStruct {
            a_string: String,
            a_opt_string_none: Option<String>,
            a_opt_string_some: Option<String>,

            t_string: String,
            t_opt_string_none: Option<String>,
            t_opt_string_some: Option<String>,

            a_u32: u32,
            a_opt_u32_none: Option<u32>,
            a_opt_u32_some: Option<u32>,

            t_u32: u32,
            t_opt_u32_none: Option<u32>,
            t_opt_u32_some: Option<u32>,

            a_i8: i8,
            a_opt_i8_none: Option<i8>,
            a_opt_i8_some: Option<i8>,

            t_i8: i8,
            t_opt_i8_none: Option<i8>,
            t_opt_i8_some: Option<i8>,

            a_bool: bool,
            a_opt_bool_none: Option<bool>,
            a_opt_bool_some: Option<bool>,

            t_bool: bool,
            t_opt_bool_none: Option<bool>,
            t_opt_bool_some: Option<bool>,

            a_f32: f32,
            a_opt_f32_none: Option<f32>,
            a_opt_f32_some: Option<f32>,

            t_f32: f32,
            t_opt_f32_none: Option<f32>,
            t_opt_f32_some: Option<f32>,
        }

        let xml = br#"
            <my-struct a_string='bla' a_opt_string_some='ble' a_u32="1" a_opt_u32_some="2" a_i8='-1' a_opt_i8_some='-2' a_bool="true" a_opt_bool_some="false" a_f32="1.1" a_opt_f32_some="2.2">
                <t_string>bli</t_string>
                <t_opt_string_some>blo</t_string_opt_some>
                <t_u32>3</t_u32>
                <t_opt_u32_some>4</t_u32_opt_some>
                <t_i8>-3</t_i8>
                <t_opt_i8_some>-4</t_i8_opt_some>
                <t_bool>false</t_bool>
                <t_opt_bool_some>true</t_bool_opt_some>
                <t_f32>3.3</t_f32>
                <t_opt_f32_some>4.4</t_f32_opt_some>
            </my-struct>"#;
        let mut parser = Parser::new(Cursor::new(&xml[..]));
        let mut deserializer = Deserializer::new(&mut parser).unwrap();

        let my_struct = MyStruct::deserialize(&mut deserializer).unwrap();
        assert_eq!(my_struct, MyStruct {
            a_string: "bla".to_string(),
            a_opt_string_none: None,
            a_opt_string_some: Some("ble".to_string()),
            t_string: "bli".to_string(),
            t_opt_string_none: None,
            t_opt_string_some: Some("blo".to_string()),
            a_u32: 1,
            a_opt_u32_none: None,
            a_opt_u32_some: Some(2),
            t_u32: 3,
            t_opt_u32_none: None,
            t_opt_u32_some: Some(4),
            a_i8: -1,
            a_opt_i8_none: None,
            a_opt_i8_some: Some(-2),
            t_i8: -3,
            t_opt_i8_none: None,
            t_opt_i8_some: Some(-4),
            a_bool: true,
            a_opt_bool_none: None,
            a_opt_bool_some: Some(false),
            t_bool: false,
            t_opt_bool_none: None,
            t_opt_bool_some: Some(true),
            a_f32: 1.1,
            a_opt_f32_none: None,
            a_opt_f32_some: Some(2.2),
            t_f32: 3.3,
            t_opt_f32_none: None,
            t_opt_f32_some: Some(4.4),
        });
    }

    /*
    #[test]
    fn test_deserializer_enum() {
        #[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
        enum MyEnum {
            VariantA {
                t_string: String,
            },
            VariantB {
                t_u32: u32,
            },
        }

        let xml_a = br#"
                <variant-a>
                    <t_string>ble</t_string>
                </variant-a>"#;

        let mut parser = Parser::new(Cursor::new(&xml_a[..]));
        let mut deserializer = Deserializer::new(&mut parser).unwrap();
        let my_enum = MyEnum::deserialize(&mut deserializer).unwrap();
        assert_eq!(my_enum, MyEnum::VariantA {
            t_string: "ble".to_string(),
        });

        let xml_b = br#"
                <variant-b>
                    <t_u32>456</t_u32>
                </variant-b>"#;

        let mut parser = Parser::new(Cursor::new(&xml_b[..]));
        let mut deserializer = Deserializer::new(&mut parser).unwrap();
        let my_enum = MyEnum::deserialize(&mut deserializer).unwrap();
        assert_eq!(my_enum, MyEnum::VariantB {
            t_u32: 456,
        });
    }*/

    #[test]
    fn test_proper() {
        #[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
        #[serde(untagged)]
        enum MyEnum {
            VariantA {
                bla: String,
            },
            VariantB {
                ble: String,
            },
        }

        #[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
        struct MyStruct {
            attr: String,

            #[serde(flatten)]
            kind: MyEnum,
        }

        let xml = br#"<variant-a attr='aaa'><bla>bbb</bla></variant-a>"#;
        let mut parser = Parser::new(Cursor::new(&xml[..]));
        let mut deserializer = Deserializer::new(&mut parser).unwrap();
        let my_struct = MyStruct::deserialize(&mut deserializer).unwrap();
        assert_eq!(my_struct, MyStruct {
            attr: "aaa".to_string(),
            kind: MyEnum::VariantA {
                bla: "bbb".to_string(),
            }
        });

        let xml = br#"<variant-b attr='aaa'><ble>bbb</ble></variant-b>"#;
        let mut parser = Parser::new(Cursor::new(&xml[..]));
        let mut deserializer = Deserializer::new(&mut parser).unwrap();
        let my_struct = MyStruct::deserialize(&mut deserializer).unwrap();
        assert_eq!(my_struct, MyStruct {
            attr: "aaa".to_string(),
            kind: MyEnum::VariantB {
                ble: "bbb".to_string(),
            }
        });
    }
}