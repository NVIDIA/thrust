
# this script will set the proper svn properties on all the files in the tree
# It pretty much requires a gnu compatible xargs (for the -r flag).  Running
# on Linux is probably the best option
# Note: use the following line if you want to remove svn:keywords
#for ext in java php xml xsl xsd wsdl properties txt htm* css js ; do find . -path '*/.svn' -prune -o  -name "*.$ext" -print0 | xargs -0  -r  svn propdel  svn:keywords ; done


find . -path '*/.svn' -prune -o  -name "*.java" -print0 | xargs -0  -r svn propset svn:eol-style native

find . -path '*/.svn' -prune -o  -name "*.php" -print0 | xargs -0  -r svn propset svn:eol-style native

find . -path '*/.svn' -prune -o  -name "*.xml" -print0 | xargs -0  -r  svn propset svn:mime-type text/xml
find . -path '*/.svn' -prune -o  -name "*.xml" -print0 | xargs -0  -r  svn propset svn:eol-style native

find . -path '*/.svn' -prune -o  -name "*.xsl" -print0 | xargs -0  -r  svn propset svn:mime-type text/xml
find . -path '*/.svn' -prune -o  -name "*.xsl" -print0 | xargs -0  -r  svn propset svn:eol-style native

find . -path '*/.svn' -prune -o  -name "*.xsd" -print0 | xargs -0  -r  svn propset svn:mime-type text/xml
find . -path '*/.svn' -prune -o  -name "*.xsd" -print0 | xargs -0  -r  svn propset svn:eol-style native

find . -path '*/.svn' -prune -o  -name "*.wsdl" -print0 | xargs -0  -r  svn propset svn:mime-type text/xml
find . -path '*/.svn' -prune -o  -name "*.wsdl" -print0 | xargs -0  -r  svn propset svn:eol-style native

find . -path '*/.svn' -prune -o  -name "*.properties" -print0 | xargs -0  -r  svn propset svn:mime-type text/plain
find . -path '*/.svn' -prune -o  -name "*.properties" -print0 | xargs -0  -r  svn propset svn:eol-style native

find . -path '*/.svn' -prune -o  -name "*.txt" -print0 | xargs -0  -r  svn propset svn:eol-style native
find . -path '*/.svn' -prune -o  -name "*.txt" -print0 | xargs -0  -r  svn propset svn:mime-type text/plain

find . -path '*/.svn' -prune -o  -name "*.htm*" -print0 | xargs -0  -r  svn propset svn:eol-style native
find . -path '*/.svn' -prune -o  -name "*.htm*" -print0 | xargs -0  -r  svn propset svn:mime-type text/html

find . -path '*/.svn' -prune -o  -name "*.css" -print0 | xargs -0  -r  svn propset svn:eol-style native
find . -path '*/.svn' -prune -o  -name "*.css" -print0 | xargs -0  -r  svn propset svn:mime-type text/css

find . -path '*/.svn' -prune -o  -name "*.js" -print0 | xargs -0  -r  svn propset svn:eol-style native
find . -path '*/.svn' -prune -o  -name "*.js" -print0 | xargs -0  -r  svn propset svn:mime-type text/javascript

find . -path '*/.svn' -prune -o  -name "README*" -print0 | xargs -0  -r  svn propset svn:eol-style native
find . -path '*/.svn' -prune -o  -name "README*" -print0 | xargs -0  -r  svn propset svn:mime-type text/plain

find . -path '*/.svn' -prune -o  -name "LICENSE*" -print0 | xargs -0  -r  svn propset svn:eol-style native
find . -path '*/.svn' -prune -o  -name "LICENSE*" -print0 | xargs -0  -r  svn propset svn:mime-type text/plain

find . -path '*/.svn' -prune -o  -name "NOTICE*" -print0 | xargs -0  -r  svn propset svn:eol-style native
find . -path '*/.svn' -prune -o  -name "NOTICE*" -print0 | xargs -0  -r  svn propset svn:mime-type text/plain

find . -path '*/.svn' -prune -o  -name "TODO*" -print0 | xargs -0  -r  svn propset svn:eol-style native
find . -path '*/.svn' -prune -o  -name "TODO*" -print0 | xargs -0  -r  svn propset svn:mime-type text/plain

find . -path '*/.svn' -prune -o  -name "KEYS*" -print0 | xargs -0  -r  svn propset svn:eol-style native
find . -path '*/.svn' -prune -o  -name "KEYS*" -print0 | xargs -0  -r  svn propset svn:mime-type text/plain

find . -path '*/.svn' -prune -o  -name "*.png" -print0 | xargs -0  -r  svn propset svn:mime-type image/png
find . -path '*/.svn' -prune -o  -name "*.gif" -print0 | xargs -0  -r  svn propset svn:mime-type image/gif
find . -path '*/.svn' -prune -o  -name "*.jpg" -print0 | xargs -0  -r  svn propset svn:mime-type image/jpeg
find . -path '*/.svn' -prune -o  -name "*.jpeg" -print0 | xargs -0  -r  svn propset svn:mime-type image/jpeg

find . -path '*/.svn' -prune -o  -name "*.fragment" -print0 | xargs -0  -r  svn propset svn:eol-style native
find . -path '*/.svn' -prune -o  -name "*.fragment" -print0 | xargs -0  -r  svn propset svn:mime-type text/xml

find . -path '*/.svn' -prune -o  -name "*.wsdd" -print0 | xargs -0  -r  svn propset svn:mime-type text/xml
find . -path '*/.svn' -prune -o  -name "*.wsdd" -print0 | xargs -0  -r  svn propset svn:eol-style native

find . -path '*/.svn' -prune -o  -name "ChangeLog*" -print0 | xargs -0  -r  svn propset svn:eol-style native
find . -path '*/.svn' -prune -o  -name "ChangeLog*" -print0 | xargs -0  -r  svn propset svn:mime-type text/plain

find . -path '*/.svn' -prune -o  -name "*.sh" -print0 | xargs -0  -r  svn propset svn:eol-style native
find . -path '*/.svn' -prune -o  -name "*.sh" -print0 | xargs -0  -r  svn propset svn:mime-type text/plain
find . -path '*/.svn' -prune -o  -name "*.sh" -print0 | xargs -0  -r  svn propset svn:executable ""

find . -path '*/.svn' -prune -o  -name "*.bat" -print0 | xargs -0  -r  svn propset svn:eol-style native
find . -path '*/.svn' -prune -o  -name "*.bat" -print0 | xargs -0  -r  svn propset svn:mime-type text/plain
find . -path '*/.svn' -prune -o  -name "*.bat" -print0 | xargs -0  -r  svn propset svn:executable ""

find . -path '*/.svn' -prune -o  -name "*.cmd" -print0 | xargs -0  -r  svn propset svn:eol-style native
find . -path '*/.svn' -prune -o  -name "*.cmd" -print0 | xargs -0  -r  svn propset svn:mime-type text/plain
find . -path '*/.svn' -prune -o  -name "*.cmd" -print0 | xargs -0  -r  svn propset svn:executable ""

find . -path '*/.svn' -prune -o  -name "INSTALL*" -print0 | xargs -0  -r  svn propset svn:eol-style native
find . -path '*/.svn' -prune -o  -name "INSTALL*" -print0 | xargs -0  -r  svn propset svn:mime-type text/plain

find . -path '*/.svn' -prune -o  -name "COPYING*" -print0 | xargs -0  -r  svn propset svn:eol-style native
find . -path '*/.svn' -prune -o  -name "COPYING*" -print0 | xargs -0  -r  svn propset svn:mime-type text/plain

find . -path '*/.svn' -prune -o  -name "NEWS*" -print0 | xargs -0  -r  svn propset svn:eol-style native
find . -path '*/.svn' -prune -o  -name "NEWS*" -print0 | xargs -0  -r  svn propset svn:mime-type text/plain

find . -path '*/.svn' -prune -o  -name "DISCLAIMER*" -print0 | xargs -0  -r  svn propset svn:eol-style native
find . -path '*/.svn' -prune -o  -name "DISCLAIMER*" -print0 | xargs -0  -r  svn propset svn:mime-type text/plain

svn propset svn:ignore -F etc/svn-ignores .
svn propset svn:ignore -F etc/svn-ignores features
svn propset svn:ignore -F etc/svn-ignores java
svn propset svn:ignore -F etc/svn-ignores java/common
svn propset svn:ignore -F etc/svn-ignores java/gadgets
svn propset svn:ignore -F etc/svn-ignores java/social-api
svn propset svn:ignore -F etc/svn-ignores java/server
svn propset svn:ignore -F etc/svn-ignores java/samples
