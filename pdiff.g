# Copyright (C) 2017  Stefan Vargyas
# 
# This file is part of Plain-Diff.
# 
# Plain-Diff is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# Plain-Diff is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with Plain-Diff.  If not, see <http://www.gnu.org/licenses/>.

#
# Grammar for plain-diff text format
#

# The grammar rules below prescribe with precision where it is
# allowing whitespaces in input text. That is that whitespaces
# in any plain-diff text are processed the way they were given
# and are not removed during text tokenization.
#
# As a result of this rule, the terminal symbols '///', '###'
# '>>>' and '<<<' are always placed right at the beginning of
# input text lines.
#
# Text lines corresponding to source lines -- those ruled by
# the non-terminal symbol 'text_line' -- are almost entirely
# identical with the source text lines. A plain-diff text
# line differs from the corresponding source text line if
# and only if the source text line starts with one of the
# following strings: "\\", "///", "###", "<<<" and ">>>".
# The association of such kind of source text lines with
# plain-diff text lines is elementary: prepend the former
# with the '\\' character for they to become the latter.

# Special notations:
# ~ stands for any sequence (possible empty) of whitespace
#     characters excepting NL ('\n'); that is any sequence
#     of SPACE (' '), FF ('\f'), HT ('\t'), VT ('\v') and
#     CR ('\r') characters;
# . stands for any character excepting NL. (This notation
#     overrides the well-established regex notation.)

plain_diff   : diff_entry*
             ;
diff_entry   : empty_line
             | comment_line
             | file_hash
             | diff_hunk
             ;
empty_line   : ~ "\n"
             ;
comment_line : "///" .* "\n"
             ;
file_hash    : "###" ~ file_name [ ~ ":" ~ hash_algo ] ~ ":" ~ hash_digest ~ "\n"
             ;
hash_algo    : "SHA" [0-9]+
             ;
hash_digest  : [a-fA-F0-9]+
             ;
diff_hunk    : insert_hunk
             | delete_hunk
             | replace_hunk
             ;
insert_hunk  : "<<<" diff_target "\n" text_lines ">>>" ~ "\n"
             ;
delete_hunk  : ">>>" diff_target "\n" text_lines ">>>" ~ "\n"
             ;
replace_hunk : ">>>" diff_target "\n" text_lines "<<<" ~ "\n" text_lines ">>>" ~ "\n"
             ;
diff_target  : ~ file_name ~ ":" ~ line_num ~
             ;
file_name    : [^ \f\t\v\r\n:]+
             ;
line_num     : [0-9]+
             ;
text_lines   : ( text_line "\n" )*
             ;
text_line    : (?<! except_text | "\n" ) .*
             | "\\" except_text .*
             ;
except_text  : "\\"
             | "///"
             | "###"
             | "<<<"
             | ">>>"
             ;


