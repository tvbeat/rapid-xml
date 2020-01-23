use std::borrow::Cow;
use std::io::Read;

use serde::de::{DeserializeSeed, IntoDeserializer, Visitor};
use serde::{forward_to_deserialize_any, Deserializer as _};

use crate::{Error, Event, Parser};
use crate::parser::DeferredString;
use inlinable_string::InlinableString;
use std::convert::TryInto;

impl serde::de::Error for Error {
    fn custom<T: std::fmt::Display>(msg: T) -> Self {
        Error::Custom(msg.to_string())
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
    pub fn new(parser: &'a mut Parser<R>) -> Self {
        // Enter the root element
        let event = parser.next().unwrap(); // TODO: No unwrap

        match event {
            Event::StartTag(tag_name) => {
                let opening_tag = tag_name.try_into().unwrap();
                Self::new_inside_tag(parser, opening_tag, false) // TODO: No unwrap
            }

            _ => {
                todo!("Handle error!")
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
    fn with_next_text<'x: 'a, F: FnOnce(DeferredString) -> Result<O, Error>, O>(&'x mut self, f: F) -> Result<O, Error> {
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
                Event::Eof => todo!("Unexpected EOF error!"),
            }
        }

        todo!("Error: No text!");
    }
}

impl<'de: 'a, 'a, R: Read> serde::Deserializer<'de> for &'a mut Deserializer<'a, R> {
    type Error = Error;

    fn deserialize_any<V: Visitor<'de>>(self, visitor: V) -> Result<<V as Visitor<'de>>::Value, Self::Error> {
        //unimplemented!()

        // TODO: Correct option?
        //self.deserialize_map(visitor)
        self.deserialize_string(visitor)
    }

    fn deserialize_bool<V: Visitor<'de>>(self, visitor: V) -> Result<<V as Visitor<'de>>::Value, Self::Error> {
        // TODO: Do we want to accept other strings in addition to "true" and "false"?
        self.with_next_text(|t| visitor.visit_bool(t.to_str()?.parse()?))
    }

    fn deserialize_i8<V: Visitor<'de>>(self, visitor: V) -> Result<<V as Visitor<'de>>::Value, Self::Error> {
        self.with_next_text(|t| visitor.visit_i8(t.to_str()?.parse()?))
    }

    fn deserialize_i16<V: Visitor<'de>>(self, visitor: V) -> Result<<V as Visitor<'de>>::Value, Self::Error> {
        self.with_next_text(|t| visitor.visit_i16(t.to_str()?.parse()?))
    }

    fn deserialize_i32<V: Visitor<'de>>(self, visitor: V) -> Result<<V as Visitor<'de>>::Value, Self::Error> {
        self.with_next_text(|t| visitor.visit_i32(t.to_str()?.parse()?))
    }

    fn deserialize_i64<V: Visitor<'de>>(self, visitor: V) -> Result<<V as Visitor<'de>>::Value, Self::Error> {
        self.with_next_text(|t| visitor.visit_i64(t.to_str()?.parse()?))
    }

    fn deserialize_u8<V: Visitor<'de>>(self, visitor: V) -> Result<<V as Visitor<'de>>::Value, Self::Error> {
        self.with_next_text(|t| visitor.visit_u8(t.to_str()?.parse()?))
    }

    fn deserialize_u16<V: Visitor<'de>>(self, visitor: V) -> Result<<V as Visitor<'de>>::Value, Self::Error> {
        self.with_next_text(|t| visitor.visit_u16(t.to_str()?.parse()?))
    }

    fn deserialize_u32<V: Visitor<'de>>(self, visitor: V) -> Result<<V as Visitor<'de>>::Value, Self::Error> {
        self.with_next_text(|t| visitor.visit_u32(t.to_str()?.parse()?))
    }

    fn deserialize_u64<V: Visitor<'de>>(self, visitor: V) -> Result<<V as Visitor<'de>>::Value, Self::Error> {
        self.with_next_text(|t| visitor.visit_u64(t.to_str()?.parse()?))
    }

    fn deserialize_f32<V: Visitor<'de>>(self, visitor: V) -> Result<<V as Visitor<'de>>::Value, Self::Error> {
        self.with_next_text(|t| visitor.visit_f32(t.to_str()?.parse()?))
    }

    fn deserialize_f64<V: Visitor<'de>>(self, visitor: V) -> Result<<V as Visitor<'de>>::Value, Self::Error> {
        self.with_next_text(|t| visitor.visit_f64(t.to_str()?.parse()?))
    }

    fn deserialize_char<V: Visitor<'de>>(self, visitor: V) -> Result<<V as Visitor<'de>>::Value, Self::Error> {
        self.with_next_text(|text| {
            let text = text.to_str()?;
            if text.len() == 1 {
                visitor.visit_char(text.chars().next().unwrap())
            } else {
                todo!("Error: More than one character!");
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
        visitor.visit_some(self) // Eh?
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
            type Error = Error;

            fn next_key_seed<K: DeserializeSeed<'de>>(&mut self, seed: K) -> Result<Option<<K as DeserializeSeed<'de>>::Value>, Self::Error> {
                loop {
                    return match self.parser.next()? {
                        Event::AttributeName(name) =>
                            seed.deserialize(name.to_str()?.into_deserializer()).map(Some),
                        Event::StartTag(name) => {
                            self.in_tag = Some(name.try_into()?);
                            seed.deserialize(name.to_str()?.into_deserializer()).map(Some)
                        }
                        Event::EndTag(_) | Event::EndTagImmediate => Ok(None),
                        Event::StartTagDone => {
                            if self.only_attributes {
                                Ok(None)
                            } else {
                                continue;
                            }
                        }
                        e => todo!("Do we need error here? {:?}", e),
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
                    Event::AttributeValue(value) =>
                        seed.deserialize(ParseDeserializer {
                            string: value.to_str()?,
                        }),
                    e => todo!("Do we need error here? {:?}", e),
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
            type Error = Error;
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
            type Error = Error;

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

// TODO: Isn't there something like this as utility in serde already?
struct ParseDeserializer<'a> {
    string: Cow<'a, str>,
}

impl<'de: 'a, 'a> serde::Deserializer<'de> for ParseDeserializer<'a> {
    type Error = Error;

    fn deserialize_any<V: Visitor<'de>>(self, visitor: V) -> Result<<V as Visitor<'de>>::Value, Self::Error> {
        match self.string {
            Cow::Owned(string) => visitor.visit_string(string),
            Cow::Borrowed(str) => visitor.visit_str(str),
        }
    }

    fn deserialize_bool<V: Visitor<'de>>(self, visitor: V) -> Result<<V as Visitor<'de>>::Value, Self::Error> {
        // TODO: Do we want to accept other strings in addition to "true" and "false"?
        visitor.visit_bool(self.string.parse()?)
    }

    fn deserialize_i8<V: Visitor<'de>>(self, visitor: V) -> Result<<V as Visitor<'de>>::Value, Self::Error> {
        visitor.visit_i8(self.string.parse()?)
    }

    fn deserialize_i16<V: Visitor<'de>>(self, visitor: V) -> Result<<V as Visitor<'de>>::Value, Self::Error> {
        visitor.visit_i16(self.string.parse()?)
    }

    fn deserialize_i32<V: Visitor<'de>>(self, visitor: V) -> Result<<V as Visitor<'de>>::Value, Self::Error> {
        visitor.visit_i32(self.string.parse()?)
    }

    fn deserialize_i64<V: Visitor<'de>>(self, visitor: V) -> Result<<V as Visitor<'de>>::Value, Self::Error> {
        visitor.visit_i64(self.string.parse()?)
    }

    fn deserialize_u8<V: Visitor<'de>>(self, visitor: V) -> Result<<V as Visitor<'de>>::Value, Self::Error> {
        visitor.visit_u8(self.string.parse()?)
    }

    fn deserialize_u16<V: Visitor<'de>>(self, visitor: V) -> Result<<V as Visitor<'de>>::Value, Self::Error> {
        visitor.visit_u16(self.string.parse()?)
    }

    fn deserialize_u32<V: Visitor<'de>>(self, visitor: V) -> Result<<V as Visitor<'de>>::Value, Self::Error> {
        visitor.visit_u32(self.string.parse()?)
    }

    fn deserialize_u64<V: Visitor<'de>>(self, visitor: V) -> Result<<V as Visitor<'de>>::Value, Self::Error> {
        visitor.visit_u64(self.string.parse()?)
    }

    fn deserialize_f32<V: Visitor<'de>>(self, visitor: V) -> Result<<V as Visitor<'de>>::Value, Self::Error> {
        visitor.visit_f32(self.string.parse()?)
    }

    fn deserialize_f64<V: Visitor<'de>>(self, visitor: V) -> Result<<V as Visitor<'de>>::Value, Self::Error> {
        visitor.visit_f64(self.string.parse()?)
    }

    fn deserialize_char<V: Visitor<'de>>(self, visitor: V) -> Result<<V as Visitor<'de>>::Value, Self::Error> {
        if self.string.len() == 1 {
            visitor.visit_char(self.string.chars().next().unwrap())
        } else {
            todo!("Error!");
        }
    }

    fn deserialize_str<V: Visitor<'de>>(self, visitor: V) -> Result<<V as Visitor<'de>>::Value, Self::Error> {
        visitor.visit_str(self.string.as_ref())
    }

    fn deserialize_string<V: Visitor<'de>>(self, visitor: V) -> Result<<V as Visitor<'de>>::Value, Self::Error> {
        visitor.visit_string(self.string.into_owned())
    }

    fn deserialize_bytes<V: Visitor<'de>>(self, visitor: V) -> Result<<V as Visitor<'de>>::Value, Self::Error> {
        visitor.visit_bytes(self.string.as_bytes())
    }

    fn deserialize_byte_buf<V: Visitor<'de>>(self, visitor: V) -> Result<<V as Visitor<'de>>::Value, Self::Error> {
        self.deserialize_bytes(visitor) // TODO: Correct?
    }

    forward_to_deserialize_any! { option unit unit_struct newtype_struct seq tuple tuple_struct map struct enum }

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
        #[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
        struct MyStruct {
            a_string: String,
            t_string: String,
            a_u32: u32,
            t_u32: u32,
            a_i8: i8,
            t_i8: i8,
            a_bool: bool,
            t_bool: bool,
        }

        let xml = br#"
            <my-struct a_string='bla' a_u32="123" a_i8='-1' a_bool="true">
                <t_string>ble</t_string>
                <t_u32>456</t_u32>
                <t_i8>-2</t_i8>
                <t_bool>false</t_bool>
            </my-struct>"#;
        let mut parser = Parser::new(Cursor::new(&xml[..]));
        let mut deserializer = Deserializer::new(&mut parser);

        let my_struct = MyStruct::deserialize(&mut deserializer).unwrap();
        assert_eq!(my_struct, MyStruct {
            a_string: "bla".to_string(),
            t_string: "ble".to_string(),
            a_u32: 123,
            t_u32: 456,
            a_i8: -1,
            t_i8: -2,
            a_bool: true,
            t_bool: false,
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
        let mut deserializer = Deserializer::new(&mut parser);
        let my_enum = MyEnum::deserialize(&mut deserializer).unwrap();
        assert_eq!(my_enum, MyEnum::VariantA {
            t_string: "ble".to_string(),
        });

        let xml_b = br#"
                <variant-b>
                    <t_u32>456</t_u32>
                </variant-b>"#;

        let mut parser = Parser::new(Cursor::new(&xml_b[..]));
        let mut deserializer = Deserializer::new(&mut parser);
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
        let mut deserializer = Deserializer::new(&mut parser);
        let my_struct = MyStruct::deserialize(&mut deserializer).unwrap();
        assert_eq!(my_struct, MyStruct {
            attr: "aaa".to_string(),
            kind: MyEnum::VariantA {
                bla: "bbb".to_string(),
            }
        });

        let xml = br#"<variant-b attr='aaa'><ble>bbb</ble></variant-b>"#;
        let mut parser = Parser::new(Cursor::new(&xml[..]));
        let mut deserializer = Deserializer::new(&mut parser);
        let my_struct = MyStruct::deserialize(&mut deserializer).unwrap();
        assert_eq!(my_struct, MyStruct {
            attr: "aaa".to_string(),
            kind: MyEnum::VariantB {
                ble: "bbb".to_string(),
            }
        });
    }
}