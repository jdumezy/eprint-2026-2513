//! The backends the shaped operations run on.

use poulpy_hal::layouts::Backend;

/// A backend whose buffers are host byte slices of `i64` limbs: every CPU backend of poulpy.
///
/// The shaped rounding pass runs on the host over the limbs of a column, so it reads and
/// writes backend views directly as slices.
pub trait HostBackend:
    'static + Backend<ZnxWord = i64> + for<'a> Backend<BufRef<'a> = &'a [u8], BufMut<'a> = &'a mut [u8]>
{
}

impl<BE> HostBackend for BE where
    BE: 'static + Backend<ZnxWord = i64> + for<'a> Backend<BufRef<'a> = &'a [u8], BufMut<'a> = &'a mut [u8]>
{
}
