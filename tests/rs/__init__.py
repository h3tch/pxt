import pxt.build
import pxt.link

lib_filename = pxt.build.rust('Cargo.toml')


@pxt.link.lib(lib_filename)
def i_add(a: int, b: int) -> int:
    print('call python i_add')
    return a + b