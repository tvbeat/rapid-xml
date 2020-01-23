use std::convert::TryInto;
use std::io::Read;
use std::marker::PhantomData;

use serde::de::DeserializeOwned;
use tuple_utils::Prepend;

use crate::{Error, Event, Parser};
use crate::de::Deserializer;

/// Utility that turns single-element tuple into that element and keeps multi-element tuples as they are
pub trait DeTuple {
    type Output;
    fn detuple(self) -> Self::Output;
}

impl<A> DeTuple for (A,) {
    type Output = A;
    fn detuple(self) -> Self::Output { self.0 }
}

macro_rules! detuple_impls {
    ($i:ident, $j:ident) => {
        detuple_impls!(impl $i, $j);
    };

    ($i:ident, $j:ident, $($r_i:ident),+) => {
        detuple_impls!(impl $i, $j, $($r_i),+);
        detuple_impls!($j, $($r_i),+);
    };

    (impl $($i:ident),+) => {
        impl<$($i),+> DeTuple for ($($i),+) {
            type Output = ($($i),+);
            fn detuple(self) -> Self::Output { self }
        }
    };
}

detuple_impls!(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P);


/// Like the try! macro (i.e. ? operator), but wraps the error in Some before returning.
macro_rules! try_some {
    ($e:expr) => {
        match $e {
            Ok(v) => v,
            Err(err) => return Some(Err(core::convert::From::from(err)))
        }
    }
}

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

impl<N: XmlPath> ElementEnter<N> {
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
pub trait XmlPath: private::Sealed {
    type Output;

    fn go<R: Read>(&mut self, parser: &mut Parser<R>) -> Option<Result<Self::Output, Error>>;
}

impl<N: XmlPath> XmlPath for ElementEnter<N> {
    type Output = N::Output;

    fn go<R: Read>(&mut self, parser: &mut Parser<R>) -> Option<Result<Self::Output, Error>> {
        loop {
            if self.entered {
                if let Some(out) = self.next.go(parser) {
                    return Some(out);
                }
            }
            self.entered = false;

            let event = try_some!(parser.next());
            match event {
                Event::StartTag(tag_name) => {
                    if self.tag == "*" || self.tag == try_some!(tag_name.to_str()) {
                        self.entered = true;
                    } else {
                        try_some!(parser.finish_tag(1));
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

impl<T: DeserializeOwned + Clone, N: XmlPath> XmlPath for ElementEnterDeserialize<T, N>
    where N::Output: Prepend<T>
{
    type Output = <N::Output as Prepend<T>>::Output;

    fn go<R: Read>(&mut self, parser: &mut Parser<R>) -> Option<Result<Self::Output, Error>> {
        loop {
            if let Some(entered) = &self.entered {
                if let Some(out) = self.next.go(parser) {
                    return match out {
                        Ok(out) => {
                            // We clone one more times than necessary (the last remaining one will get dropped once
                            // the `next` returns `None`. It would be nice if we could avoid the last clone, but
                            // we would have to know that the underlying `next` really returned the last one.
                            Some(Ok(out.prepend((*entered).clone())))
                        }
                        Err(err) => Some(Err(err))
                    };
                }
            }
            self.entered = None;

            let event = try_some!(parser.next());
            match event {
                Event::StartTag(tag_name) => {
                    if self.tag == "*" || self.tag == try_some!(tag_name.to_str()) {
                        let opening_tag = try_some!(tag_name.try_into());
                        let mut des = Deserializer::new_inside_tag(parser, opening_tag, true);
                        self.entered = Some(try_some!(T::deserialize(&mut des)));
                    } else {
                        try_some!(parser.finish_tag(1));
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

impl<T: DeserializeOwned> XmlPath for ElementDeserialize<T> {
    type Output = (T,);

    fn go<R: Read>(&mut self, parser: &mut Parser<R>) -> Option<Result<Self::Output, Error>> {
        loop {
            let event = try_some!(parser.next());
            match event {
                Event::StartTag(tag_name) => {
                    if self.tag == "*" || self.tag == try_some!(tag_name.to_str()) {
                        let opening_tag = try_some!(tag_name.try_into());
                        let mut des = Deserializer::new_inside_tag(parser, opening_tag, false);
                        return Some(Ok((try_some!(T::deserialize(&mut des)), )))
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
/// See `xml_path` macro for easy way of constructing the path.
///
/// TreeDeserializer is an `Iterator` over the deserialized elements. If the xml path deserializes
/// exactly one type, the iterator will yield that type. If the xml path deserializes multiple
/// types (nested in a XML tree), the iterator will yield tuple of those types.
///
/// TODO: Example
pub struct TreeDeserializer<R: Read, N> {
    parser: Parser<R>,
    path: N,
}

impl<R: Read, N: XmlPath> TreeDeserializer<R, N> {
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

impl<R: Read, N: XmlPath> Iterator for TreeDeserializer<R, N> where N::Output: DeTuple {
    type Item = Result<<N::Output as DeTuple>::Output, Error>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.path.go(&mut self.parser) {
            Some(Ok(tuple)) => Some(Ok(tuple.detuple())),
            Some(Err(err)) => Some(Err(err)),
            None => None,
        }
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

    const SAMPLE_XML: &[u8] = br#"
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

    const SAMPLE_XML_ERRORS: &[u8] = br#"
            <root>
                <aaa>
                    <bbb n="1">
                        <ccc m="100"/>
                        <ccc/>
                        <ccc m="200"/>
                        <ccc m=250/>
                    </bbb>
                    <xxx>Unknown tag</xxx>
                </aaa>
                <xxx>Unknown tag</xxx>
                <aaa>
                    <bbb n="99">Matched tag without anything nested</bbb>
                    <bbb n="2">
                        <ccc><m>300</m></ccc>
                        <ccc><m>asdf</m></ccc>
                    </bbb>
                </aaa>
            </root>
        "#;

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
        let path = xml_path!("root", "aaa", "bbb", "ccc" => Ccc);

        let mut des = TreeDeserializer::from_reader(path, Cursor::new(&SAMPLE_XML[..]));

        assert_eq!(des.next().unwrap().unwrap(), Ccc { m: 100 });
        assert_eq!(des.next().unwrap().unwrap(), Ccc { m: 200 });
        assert_eq!(des.next().unwrap().unwrap(), Ccc { m: 300 });
        assert_eq!(des.next().unwrap().unwrap(), Ccc { m: 400 });
        assert!(des.next().is_none());
    }

    #[test]
    fn multiple_elements() {
        let path = xml_path!("root", "aaa", "bbb" => Bbb, "ccc" => Ccc);

        let mut des = TreeDeserializer::from_reader(path, Cursor::new(&SAMPLE_XML[..]));

        assert_eq!(des.next().unwrap().unwrap(), (Bbb { n: 1 }, Ccc { m: 100 }));
        assert_eq!(des.next().unwrap().unwrap(), (Bbb { n: 1 }, Ccc { m: 200 }));
        assert_eq!(des.next().unwrap().unwrap(), (Bbb { n: 2 }, Ccc { m: 300 }));
        assert_eq!(des.next().unwrap().unwrap(), (Bbb { n: 2 }, Ccc { m: 400 }));
        assert!(des.next().is_none());
    }

    #[test]
    fn with_errors() {
        let path = xml_path!("root", "aaa", "bbb" => Bbb, "ccc" => Ccc);

        let mut des = TreeDeserializer::from_reader(path, Cursor::new(&SAMPLE_XML_ERRORS[..]));

        // TODO: The actual output may still change if we learn to recover from some errors better.

        assert_eq!(des.next().unwrap().unwrap(), (Bbb { n: 1 }, Ccc { m: 100 }));
        assert!(des.next().unwrap().is_err());
        assert_eq!(des.next().unwrap().unwrap(), (Bbb { n: 1 }, Ccc { m: 200 }));
        assert!(des.next().unwrap().is_err()); // Bad attribute value
        assert!(des.next().unwrap().is_err()); // Bad attribute name (rest of the bad value)
        assert_eq!(des.next().unwrap().unwrap(), (Bbb { n: 2 }, Ccc { m: 300 }));
        assert!(des.next().unwrap().is_err());
        assert!(des.next().is_none());
    }
}
