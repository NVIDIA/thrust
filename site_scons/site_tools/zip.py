"""SCons.Tool.zip

Tool-specific initialization for zip.

There normally shouldn't be any need to import this module directly.
It will usually be imported through the generic SCons.Tool.Tool()
selection method.

This version applies the patch from scons.tigris.org/issues/show_bug.cgi?id=2575

"""

#
# Copyright (c) 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010 The SCons Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY
# KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#

__revision__ = "src/engine/SCons/Tool/zip.py 5134 2010/08/16 23:02:40 bdeegan"

import os.path

import SCons.Builder
import SCons.Defaults
import SCons.Node.FS
import SCons.Util

try:
    import zipfile
    internal_zip = 1
except ImportError:
    internal_zip = 0

if internal_zip:
    zipcompression = zipfile.ZIP_DEFLATED
    def zip(target, source, env):
        compression = env.get('ZIPCOMPRESSION', 0)
        zf = zipfile.ZipFile(target[0].abspath, 'w', compression)
        for s in source:
            if s.isdir():
                for dirpath, dirnames, filenames in os.walk(os.path.relpath(s.abspath)):
                    for fname in filenames:
                        path = os.path.join(dirpath, fname)
                        if os.path.isfile(path):
                            zf.write(path)
            else:
                zf.write(os.path.relpath(s.abspath))
        zf.close()
else:
    zipcompression = 0
    zip = "$ZIP $ZIPFLAGS ${TARGET.abspath} $SOURCES"


zipAction = SCons.Action.Action(zip, varlist=['ZIPCOMPRESSION'])

ZipBuilder = SCons.Builder.Builder(action = SCons.Action.Action('$ZIPCOM', '$ZIPCOMSTR'),
                                   source_factory = SCons.Node.FS.Entry,
                                   source_scanner = SCons.Defaults.DirScanner,
                                   suffix = '$ZIPSUFFIX',
                                   multi = 1)


def generate(env):
    """Add Builders and construction variables for zip to an Environment."""
    try:
        bld = env['BUILDERS']['Zip']
    except KeyError:
        bld = ZipBuilder
        env['BUILDERS']['Zip'] = bld

    env['ZIP']        = 'zip'
    env['ZIPFLAGS']   = SCons.Util.CLVar('')
    env['ZIPCOM']     = zipAction
    env['ZIPCOMPRESSION'] =  zipcompression
    env['ZIPSUFFIX']  = '.zip'

def exists(env):
    return internal_zip or env.Detect('zip')

# Local Variables:
# tab-width:4
# indent-tabs-mode:nil
# End:
# vim: set expandtab tabstop=4 shiftwidth=4:
