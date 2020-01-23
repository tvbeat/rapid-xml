use std::io::Read;
use std::marker::PhantomData;

use serde::de::DeserializeOwned;
use tuple_utils::Prepend;

use crate::{Event, Parser};
use crate::de::Deserializer;
use std::convert::TryInto;

// TODO: Remove all unwraps!

/// Part of path for `TreeDeserializer`.
///
/// Enters matched element in a tree, nothing is deserialized.
///
/// You may want to use the `xml_path!` macro rather than constructing path manually.
#[derive(Debug)]
pub struct ElementEnter<N> {
    tag: &'static str,
    next: N,

    entered: bool,
}

impl<N: NextStep> ElementEnter<N> {
    /// Create `ElementEnter` matching given tag
    pub fn tag(tag: &'static str, next: N) -> Self {
        Self { tag, next, entered: false }
    }

    /// Create `ElementEnter` matching any tag
    pub fn any(next: N) -> Self {
        Self { tag: "*", next, entered: false }
    }
}

impl<N: PartialEq<N>> PartialEq<ElementEnter<N>> for ElementEnter<N> {
    fn eq(&self, other: &ElementEnter<N>) -> bool {
        self.tag == other.tag && self.next == other.next
        // State is ignored
    }
}

/// Part of path for `TreeDeserializer`.
///
/// Enters matched element in a tree. It's attributes are deserialized into given Deserializable
/// type and the deserializer proceeds with contained elements. The type must be `Clone` because it
/// may be returned multiple times if there are multiple contained elements.
///
/// You may want to use the `xml_path!` macro rather than constructing path manually.
#[derive(Debug)]
pub struct ElementEnterDeserialize<T, N> {
    tag: &'static str,
    next: N,

    entered: Option<T>,
}

impl<T, N> ElementEnterDeserialize<T, N> {
    /// Create `ElementEnterDeserialize` matching given tag
    pub fn tag(tag: &'static str, next: N) -> Self {
        Self { tag, next, entered: None }
    }

    /// Create `ElementEnterDeserialize` matching any tag
    pub fn any(next: N) -> Self {
        Self { tag: "*", next, entered: None }
    }
}

impl<T: DeserializeOwned, N: PartialEq<N>> PartialEq<ElementEnterDeserialize<T, N>> for ElementEnterDeserialize<T, N> {
    fn eq(&self, other: &ElementEnterDeserialize<T, N>) -> bool {
        self.tag == other.tag && self.next == other.next
        // State is ignored
    }
}

/// Part of path for `TreeDeserializer`.
///
/// Deserializes matched element. This is the final component of every path. The matched element and
/// all nested elements are deserialized into given type.
///
/// You may want to use the `xml_path!` macro rather than constructing path manually.
#[derive(Debug, PartialEq)]
pub struct ElementDeserialize<T: DeserializeOwned> {
    tag: &'static str,
    _phantom: PhantomData<T>,
}

impl<T: DeserializeOwned> ElementDeserialize<T> {
    /// Create `ElementDeserialize` matching given tag
    pub fn tag(tag: &'static str) -> Self {
        Self { tag, _phantom: PhantomData }
    }

    /// Create `ElementDeserialize` matching any tag
    pub fn any() -> Self {
        Self { tag: "*", _phantom: PhantomData }
    }
}

mod private {
    use serde::de::DeserializeOwned;

    pub trait Sealed {}

    impl<N> Sealed for super::ElementEnter<N> {}
    impl<T: DeserializeOwned, N> Sealed for super::ElementEnterDeserialize<T, N> {}
    impl<T: DeserializeOwned> Sealed for super::ElementDeserialize<T> {}
}

#[doc(hidden)]
pub trait NextStep: private::Sealed {
    type Output;

    fn go<R: Read>(&mut self, parser: &mut Parser<R>) -> Option<Self::Output>;
}

impl<N: NextStep> NextStep for ElementEnter<N> {
    type Output = N::Output;

    fn go<R: Read>(&mut self, parser: &mut Parser<R>) -> Option<Self::Output> {
        loop {
            if self.entered {
                if let Some(out) = self.next.go(parser) {
                    return Some(out);
                }
            }
            self.entered = false;

            let event = parser.next().unwrap();
            match event {
                Event::StartTag(tag_name) => {
                    if self.tag == "*" || self.tag == tag_name.to_str().unwrap() {
                        self.entered = true;
                    } else {
                        parser.finish_tag(1).unwrap();
                    }
                },
                Event::EndTagImmediate | Event::EndTag(_) | Event::Eof => {
                    return None;
                },
                Event::StartTagDone | Event::AttributeName(_) | Event::AttributeValue(_) | Event::Text(_) => {}
            }
        }
    }
}

impl<T: DeserializeOwned + Clone, N: NextStep> NextStep for ElementEnterDeserialize<T, N>
    where N::Output: Prepend<T>
{
    type Output = <N::Output as Prepend<T>>::Output;

    fn go<R: Read>(&mut self, parser: &mut Parser<R>) -> Option<Self::Output> {
        loop {
            if let Some(entered) = &self.entered {
                if let Some(out) = self.next.go(parser) {
                    // We clone one more times than necessary (the last remaining one will get dropped once
                    // the `next` returns `None`. It would be nice if we could avoid the last clone, but
                    // we would have to know that the underlying `next` really returned the last one.
                    return Some(out.prepend((*entered).clone()))
                }
            }
            self.entered = None;

            let event = parser.next().unwrap();
            match event {
                Event::StartTag(tag_name) => {
                    if self.tag == "*" || self.tag == tag_name.to_str().unwrap() {
                        let opening_tag = tag_name.try_into().unwrap();
                        let mut des = Deserializer::new_inside_tag(parser, opening_tag, true);
                        self.entered = Some(T::deserialize(&mut des).unwrap());
                    } else {
                        parser.finish_tag(1).unwrap();
                    }
                },
                Event::EndTagImmediate | Event::EndTag(_) => {
                    return None;
                },
                Event::Eof => {
                    todo!("Error: Premature EOF");
                },
                Event::StartTagDone | Event::AttributeName(_) | Event::AttributeValue(_) | Event::Text(_) => {}
            }
        }
    }
}

impl<T: DeserializeOwned> NextStep for ElementDeserialize<T> {
    type Output = (T,);

    fn go<R: Read>(&mut self, parser: &mut Parser<R>) -> Option<Self::Output> {
        loop {
            let event = parser.next().unwrap();
            match event {
                Event::StartTag(tag_name) => {
                    if self.tag == "*" || self.tag == tag_name.to_str().unwrap() {
                        let opening_tag = tag_name.try_into().unwrap();
                        let mut des = Deserializer::new_inside_tag(parser, opening_tag, false);
                        return Some((T::deserialize(&mut des).unwrap(), ))
                    }
                },
                Event::EndTagImmediate | Event::EndTag(_) => {
                    return None;
                },
                Event::Eof => {
                    todo!("Error: Premature EOF");
                },
                Event::StartTagDone | Event::AttributeName(_) | Event::AttributeValue(_) | Event::Text(_) => {}
            }
        }
    }
}

/// Deserializer for a sequence of elements in a tree.
///
/// TODO: Example
pub struct TreeDeserializer<R: Read, N> {
    parser: Parser<R>,
    path: N,
}

impl<R: Read, N: NextStep> TreeDeserializer<R, N> {
    /// Create new `TreeDeserializer` from given `Parser`.
    pub fn new(path: N, parser: Parser<R>) -> Self {
        Self {
            parser,
            path,
        }
    }

    /// Create new `TreeDeserializer` from given IO `Read`.
    pub fn from_reader(path: N, reader: R) -> Self {
        Self::new(path, Parser::new(reader))
    }
}

impl<R: Read, N: NextStep> Iterator for TreeDeserializer<R, N> {
    type Item = N::Output; // TODO: Maybe flatten here?

    fn next(&mut self) -> Option<Self::Item> {
        self.path.go(&mut self.parser)
    }
}

/// Macro for easier construction of XML path.
///
/// This macro is alternative to building path by manually creating and nesting `ElementEnter`,
/// `ElementDeserialize` and `ElementEnterDeserialize`.
///
/// You can use "*" to match any tag. Partial matching is currently not supported!
///
/// The last element must deserialize into a type!
///
/// # Example
///
/// ```no_run
/// # use serde_derive::Deserialize;
/// # #[derive(Clone, Deserialize)]
/// # struct Ccc {}
/// # #[derive(Deserialize)]
/// # struct Eee {}
/// # use rapid_xml::xml_path;
/// let path = xml_path!("aaa", "*", "ccc" => Ccc, "*", "eee" => Eee);
/// ```
///
/// This will enter tags `<aaa ...>`, inside enter any tag, inside enter and deserialize `<ccc ...>`
/// tag into `Ccc` type, enter any tag and finaly deserialize `<eee ...` into `Eee` type.
///
/// The `TreeDeserializer` with this path will be `Iterator` over `Result<(Ccc, Eee), _>` type.
#[macro_export]
macro_rules! xml_path {
    // Tail rule - turns `"ccc" => Ccc` expression into `ElementDeserialize`
    ($tag_name:literal => $t:ty) => {
        $crate::ElementDeserialize::<$t>::tag($tag_name)
    };

    // Tail rule - to inform user that the last expression must be `"ccc" => Ccc`, can't be just
    // `"ccc"`.
    ($tag_name:literal) => {
        compile_error!("Paths must end with `\"tag_name\" => Type` expression.")
    };

    // Recursive rule - turn `"ccc" => Ccc` expression at beginning into `ElementEnterDeserialize`
    // and call ourselves recursively on the rest.
    ($tag_name:literal => $t:ty, $($r_tag_name:literal $(=> $r_t:ty)?),+) => {
        $crate::ElementEnterDeserialize::<$t, _>::tag($tag_name,
            xml_path!($($r_tag_name $(=> $r_t)?),+)
        )
    };


    // Recursive rule - turn `"ccc"` expression at beginning into `ElementEnter` and call ourselves
    // recursively on the rest.
    ($tag_name:literal, $($r_tag_name:literal $(=> $r_t:ty)?),+) => {
        $crate::ElementEnter::tag($tag_name,
            xml_path!($($r_tag_name $(=> $r_t)?),+)
        )
    };
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use serde_derive::Deserialize;

    use super::*;

    #[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
    struct Bbb {
        n: u32,
    }

    #[derive(Debug, Deserialize, PartialEq, Eq)]
    struct Ccc {
        m: u32,
    }

    #[test]
    fn xml_path_macro() {
        let path = xml_path!("bbb" => Bbb);
        assert_eq!(path, ElementDeserialize::<Bbb>::tag("bbb"));

        let path = xml_path!("bbb" => Bbb, "ccc" => Ccc);
        assert_eq!(path, ElementEnterDeserialize::<Bbb, _>::tag("bbb", ElementDeserialize::<Ccc>::tag("ccc")));

        let path = xml_path!("bbb", "ccc" => Ccc);
        assert_eq!(path, ElementEnter::tag("bbb", ElementDeserialize::<Ccc>::tag("ccc")));

        let path = xml_path!("aaa", "bbb" => Bbb, "ccc" => Ccc);
        assert_eq!(path, ElementEnter::tag("aaa", ElementEnterDeserialize::<Bbb, _>::tag("bbb", ElementDeserialize::<Ccc>::tag("ccc"))));

        let path = xml_path!("bbb" => Bbb, "aaa", "ccc" => Ccc);
        assert_eq!(path, ElementEnterDeserialize::<Bbb, _>::tag("bbb", ElementEnter::tag("aaa", ElementDeserialize::<Ccc>::tag("ccc"))));
    }

    #[test]
    fn basic() {
        let xml = br#"
            <root>
                <aaa>
                    <bbb n="1">
                        <ccc m="100"/>
                        <ccc m="200"/>
                    </bbb>
                    <xxx>Unknown tag</xxx>
                </aaa>
                <xxx>Unknown tag</xxx>
                <aaa>
                    <bbb n="99">Matched tag without anything nested</bbb>
                    <bbb n="2">
                        <ccc><m>300</m></ccc>
                        <ccc><m>400</m></ccc>
                    </bbb>
                </aaa>
            </root>
        "#;

        let path = xml_path!("root", "aaa", "bbb" => Bbb, "ccc" => Ccc);

        let mut hir_des = TreeDeserializer::from_reader(path, Cursor::new(&xml[..]));

        assert_eq!(hir_des.next(), Some((Bbb { n: 1 }, Ccc { m: 100 })));
        assert_eq!(hir_des.next(), Some((Bbb { n: 1 }, Ccc { m: 200 })));
        assert_eq!(hir_des.next(), Some((Bbb { n: 2 }, Ccc { m: 300 })));
        assert_eq!(hir_des.next(), Some((Bbb { n: 2 }, Ccc { m: 400 })));
        assert_eq!(hir_des.next(), None);
    }
}