import pxt.build
import pxt.link
from pxt.cuda import In, Out


@pxt.link.cuda('multiply.cubin')
@pxt.build.cuda('multiply.cu')
def multiply(out: Out, a: In, b: In, **kwargs):
    pass
