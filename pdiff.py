#!/usr/bin/python

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

import sys, os, re

from types import NoneType
from itertools import imap, repeat

program = 'pdiff'
verdate = '0.2 2017-12-19 13:35' # $ date +'%F %R'

license = """\
Copyright (C) 2017  Stefan Vargyas.
License GPLv3+: GNU GPL version 3 or later <http://gnu.org/licenses/gpl.html>.
This is free software: you are free to change and redistribute it.
There is NO WARRANTY, to the extent permitted by law."""

def joinln(args):
    return "\n".join(imap(str, args))

def write(where, msg, args):
    s = isinstance(msg, str)
    l = isinstance(msg, list) or \
        isinstance(msg, tuple)
    n = len(args)
    if l:
        msg = joinln(msg)
    elif not s:
        msg = str(msg)
    if s and n:
        msg = msg % args
    elif not s and n:
        msg += "\n" + joinln(args)
    where.write(msg)

def cout(msg = "", *args):
    write(sys.stdout, msg, args)

def cerr(msg = "", *args):
    write(sys.stderr, msg, args)

def error(msg, *args):
    if len(args):
        msg = msg % args
    cerr("%s: error: %s\n", program, msg)
    sys.exit(1)

def warn(msg, *args):
    if len(args):
        msg = msg % args
    cerr("%s: warning: %s\n", program, msg)

def int_bits():
    from math import log
    return int(log(sys.maxint, 2)) + 1

class Hash:

    @staticmethod
    def constr(algo):
        import hashlib
        try:
            return getattr(hashlib, algo.lower())()
        except:
            error("failed creating hash object '%s'", algo)

    def __init__(self, algo):
        self.algo = Hash.constr(algo)
        self.name = algo

    def digest_size(self):
        return self.algo.digest_size * 2

    def digest(self, file):
        assert isinstance(file, File)
        assert file.read_lines()
        for l in file:
            self.algo.update(l)
        return self.algo.hexdigest()

class File:

    @staticmethod
    def uid(name):
        try:
            s = os.stat(name)
            return s.st_ino, s.st_dev
        except:
            return None

    @staticmethod
    def read(name, func = file.read):
        try:
            f = open(name)
        except:
            error("failed opening file '%s'", name)
        try:
            return func(f)
        except:
            error("failed reading file '%s'", name)
        finally:
            f.close()

    def __init__(self, name, dry_run):
        assert isinstance(name, str)
        if not dry_run:
            self.uid = File.uid(name) or name
        else:
            self.uid = name
        self.lines = None
        self.name = name

    def exists(self):
        return isinstance(self.uid, tuple)

    def read_lines(self):
        if self.exists() and self.lines is None:
            self.lines = File.read(
                self.name, file.readlines)
        return self.lines is not None

    def same_text(self, lines, start, result = None):
        assert isinstance(result, (NoneType, set))
        assert start > 0
        if result is not None:
            result.clear()

        r = 0
        i = 0
        j = start - 1
        n = len(lines)
        m = len(self.lines)
        while i < n:
            if j >= m or lines[i] != self.lines[j]:
                if result is not None:
                    result.add(i + 1)
                else:
                    return False
                r += 1
            i += 1
            j += 1

        return r == 0

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        return self.lines[index]

    def __iter__(self):
        return iter(self.lines)

    def __eq__(self, file):
        return \
            isinstance(file, File) and \
            self.uid == file.uid

    def __ne__(self, file):
        return not self.__eq__(file)

    def __lt__(self, file):
        assert isinstance(file, File)
        return self.uid < file.uid

    def __gt__(self, file):
        assert isinstance(file, File)
        return self.uid > file.uid

    def __hash__(self):
        return hash(self.uid)

    def __str__(self):
        return str(self.name)

    def __repr__(self):
        return repr(self.name)

    def not_impl(self, method):
        error("method 'File.%s' not implemented",
            method) 

    __le__ = lambda self, arg: self.not_impl("__le__")
    __ge__ = lambda self, arg: self.not_impl("__ge__")

class PlainHash:

    def __init__(self, file, algo, digest, pos):
        self.file = file
        self.algo = algo
        self.digest = digest
        self.pos = pos

    def write(self, where):
        h = str(self.file)
        if self.algo is not None:
            h += ":%s" % self.algo
        where("### %s:%s\n", h, self.digest)

class PlainDiff:

    def __init__(self, file, line, addr, pos):
        self.file = file
        self.line = line
        self.addr = addr
        self.pos = pos
        self.source = []
        self.target = []

    def check_text(self, result = None):
        assert isinstance(result, (NoneType, set))
        assert self.file.read_lines()
        return self.file.same_text(
            self.source,
            self.line,
            result)

    @staticmethod
    def is_except(text):
        if len(text):
            c = text[0]
            return c == '\\' or \
                c in "/#<>" and \
                text.startswith(c * 3)
        else:
            return False

    def write(self, where):
        s = len(self.source)
        t = len(self.target)
        assert s or t
        if s:
            where(">>>")
        else:
            where("<<<")
        if True:
            where(" %s:%d\n", 
                self.file,
                self.line)
        for l in self.source:
            not self.is_except(l) or \
            where("\\")
            where(l)
        if s and t:
            where("<<<\n")
        for l in self.target:
            not self.is_except(l) or \
            where("\\")
            where(l)
        if True:
            where(">>>\n")

    @staticmethod
    def cmp_file_line(a, b):
        if a.file == b.file:
            return a.line - b.line
        if a.file < b.file:
            return -1
        else:
            return 1

    @staticmethod
    def cmp_file(a, b):
        if a.file == b.file:
            return 0
        if a.file < b.file:
            return -1
        else:
            return 1

class Plain:

    def __init__(self, hashes, diffs):
        self.hashes = hashes
        self.diffs = diffs

    def write(self, where):
        for f in sorted(self.hashes.keys()):
            self.hashes[f].write(where)
        for d in self.diffs:
            d.write(where)

class Gen:

    def __init__(self, diffs, opt):
        self.diffs = diffs
        self.verbose = opt.verbose

    def write(self, where):
        for d in self.diffs:
            for l in self.gen(d):
                where(l)

class Unified(Gen):

    def __init__(self, diffs, opt):
        Gen.__init__(self, diffs, opt)
        self.context = opt.context
        self.no_adjust = opt.dry_run or \
            opt.no_adjust
        self.current = None
        self.hunkno = None
        if not opt.no_reorder:
            c = PlainDiff.cmp_file_line
        else:
            c = PlainDiff.cmp_file
        self.diffs.sort(c)

    def gen(self, diff):
        from difflib import SequenceMatcher

        if not self.no_adjust:
            assert diff.file.read_lines()

        if self.verbose:
            yield "# pdiff: file=%r line=%d pos=%d\n" % (
                diff.file,
                diff.line,
                diff.pos)

        if self.current != diff.file:
            self.current = diff.file
            self.hunkno = 0
            yield "--- %s\n" % diff.file
            yield "+++ %s\n" % diff.file

        s = SequenceMatcher(
                None, diff.source, diff.target)
        for g in s.get_grouped_opcodes(self.context):
            i1 = g[0][1]
            i2 = g[-1][2]
            j1 = g[0][3]
            j2 = g[-1][4]

            assert i2 >= i1
            assert j2 >= j1

            if self.no_adjust:
                k1 = 0
            elif g[0][0] != 'equal':
                k1 = self.context
            else:
                k1 = self.context - \
                    (g[0][2] - g[0][1])
                assert k1 >= 0

            if self.no_adjust:
                k2 = 0
            elif g[-1][0] != 'equal':
                k2 = self.context
            else:
                k2 = self.context - \
                    (g[-1][2] - g[-1][1])
                assert k2 >= 0

            if not self.no_adjust:
                if k1 >= i1 + diff.line:
                    k1 = i1 + diff.line - 1

                n = len(diff.file)
                assert n >= i2 + diff.line - 1
                if k2 >= n - i2 - diff.line + 1:
                    k2 = n - i2 - diff.line + 1

            self.hunkno += 1
            yield "@@ -%d,%d +%d,%d @@ #%d\n" % (
                i1 + diff.line - k1,
                i2 - i1 + k1 + k2,
                j1 + diff.line - k1,
                j2 - j1 + k1 + k2,
                self.hunkno
            )

            for i in range(
                i1 + diff.line - 1 - k1,
                i1 + diff.line - 1):
                yield ' ' + diff.file[i]

            for t, i1, i2, j1, j2 in g:
                if t == 'equal':
                    for l in diff.source[i1:i2]:
                        yield ' ' + l
                    continue
                if t == 'replace' or t == 'delete':
                    for l in diff.source[i1:i2]:
                        yield '-' + l
                if t == 'replace' or t == 'insert':
                    for l in diff.target[j1:j2]:
                        yield '+' + l

            for i in range(
                i2 + diff.line - 1,
                i2 + diff.line - 1 + k2):
                yield ' ' + diff.file[i]

class TTY:

    def __init__(self):
        fd = os.open("/dev/tty", os.O_RDWR|os.O_NOCTTY)
        self.tty = os.fdopen(fd, 'w+', 1)

    def write(self, text):
        self.tty.write(text)
        self.tty.flush()

    def input(self, prompt):
        self.write(prompt)
        try:
            l = self.tty.readline()
        except KeyboardInterrupt:
            self.write("\n")
            self.close()
            sys.exit(0)
        if not l:
            self.write("^D\n")
            raise EOFError
        if l[-1] == '\n':
            l = l[:-1]
        return l

    def close(self):
        try:
            self.tty.close()
        except:
            pass

# Sedgewick & Wayne: Algorithms 4th ed, p. 768
# Algo. 5.6, Knuth-Morris-Pratt substring search

class KMP:

    class Symbol:

        SIZE = 256

        def __init__(self, seq):
            self.seq = seq

        def __call__(self, index):
            return ord(self.seq[index])

        def __getitem__(self, index):
            return self.seq[index]

        def __len__(self):
            return len(self.seq)

    def __init__(self, pattern, symbol = Symbol):
        assert issubclass(symbol, self.Symbol)
        assert isinstance(pattern, symbol)
        self.pattern = pattern
        self.symbol = symbol

        r = self.symbol.SIZE
        m = len(self.pattern)
        assert m < 256
        assert m > 0

        from numpy import zeros, uint8
        self.dfa = zeros(shape = (r, m), dtype = uint8)
        self.dfa[self.pattern(0)][0] = 1

        x = 0
        for j in xrange(1, m):
            for c in xrange(r):
                self.dfa[c][j] = self.dfa[c][x]
            self.dfa[self.pattern(j)][j] = j + 1
            x = self.dfa[self.pattern(j)][x]

    def search_at(self, text, pos):
        assert isinstance(text, self.symbol)

        m = len(self.pattern)
        n = len(text)

        assert pos >= 0
        assert pos < n
        assert m < n

        j = 0
        i = pos
        while i < n and j < m:
            j = self.dfa[text(i)][j]
            i += 1

        if j == m:
            # found (hit end of pattern)
            return i - m
        else:
            # not found (hit end of text)
            return n

    def same_syms(self, text, pos):
        m = len(self.pattern)
        n = len(text)

        assert pos >= 0
        assert pos + m <= n
        assert m < n

        for i in xrange(m):
            if self.pattern[i] != text[pos + i]:
                return False
        return True

    def search(self, text):
        assert isinstance(text, self.symbol)

        m = len(self.pattern)
        n = len(text)
        r = []

        i = 0
        while i < n:
            j = self.search_at(text, i)
            assert j <= n
            assert j >= i

            if j >= n:
                break
            if self.same_syms(text, j):
                r.append(j + 1)
                i = j + m
            else:
                i = j + 1

        return r

class Addresses(Gen):

    class Symbol(KMP.Symbol):

        BITS = 9
        SHIFT = int_bits() - BITS
        SIZE = 1 << BITS

        def __init__(self, seq):
            KMP.Symbol.__init__(self, seq)
            self.hashes = list(repeat(None, len(seq)))
            assert self.SHIFT > 0

        def __call__(self, index):
            r = self.hashes[index]
            if r is None:
                r = abs(hash(self.seq[index]) >> self.SHIFT)
                assert r < self.SIZE
                self.hashes[index] = r
            return r

    def __init__(self, diffs, opt):
        Gen.__init__(self, diffs, opt)
        self.name = opt.input or '<stdin>'
        self.no_hunk_out = opt.no_hunk_out
        self.interactive = opt.interactive and \
            not self.no_hunk_out
        if self.no_hunk_out:
            self.error = self.cout
        self.current = -1
        self.n_hunks = 0

    def error(self, pos, msg, *args):
        error("%s:%d: %s", self.name, pos, msg % args)

    def cout(self, pos, msg, *args):
        cout("%s:%d: %s", self.name, pos, msg % args)
        cout("\n")

    def interact(self, choices, prompt):
        c = map(str, choices)
        p = "%s [%s]: " % (prompt,
            ", ".join(c))
        t = TTY()
        try:
            while True:
                try:
                    r = t.input(p)
                except EOFError:
                    continue
                if r.strip() in c:
                    return int(r)
        finally:
            t.close()

    def symbol(self, file):
        try:
            return file.symbol
        except AttributeError:
            file.symbol = self.Symbol(file)
            return file.symbol

    def gen(self, diff):
        self.current += 1

        if not len(diff.source):
            return

        assert diff.file.read_lines()

        if not self.no_hunk_out and \
            diff.check_text():
            return

        m = KMP(
            symbol = self.Symbol,
            pattern = self.Symbol(diff.source))
        t = m.search(self.symbol(diff.file))
        assert isinstance(t, list)

        if self.verbose:
            cerr("diff-hunk: %2.0f%%\r",
                100 * float(self.current) /
                len(self.diffs))

        # stev: interactive => output hunks
        assert not self.interactive or \
            not self.no_hunk_out

        n = len(t)
        if n == 0:
            self.error(diff.pos,
                "diff hunk source lines "
                "not found in source text")
        elif n == 1 and diff.line == t[0]:
            assert self.no_hunk_out
        elif n == 1 and self.no_hunk_out:
            self.error(diff.pos,
                "diff hunk source lines occur "
                "somewhere else in source text: %s",
                t[0])
        elif n > 1 and not self.interactive:
            self.error(diff.pos,
                "diff hunk source lines occur "
                "multiple times in source text: %s",
                ", ".join(map(str, t)))
        elif n > 1:
            t = self.interact(t,
                "%s:%d: %s: source lines "
                "matching choice" % (
                self.name,
                diff.pos,
                diff.file))
            assert isinstance(t, int)
        else:
            assert not self.no_hunk_out
            assert n == 1
            t = t[0]

        if self.no_hunk_out:
            return

        if not self.interactive and \
            diff.line != t and \
            self.n_hunks:
            yield '\n'

        if diff.line != t:
            self.n_hunks += 1
            l = '\\' + diff.addr

            yield '>>> %s:%d\n' % (self.name, diff.pos)
            yield l % (diff.file, diff.line)
            yield '<<<\n'
            yield l % (diff.file, t)
            yield '>>>\n'

class Parser:

    def __init__(self, opt):
        self.input = opt.input
        self.no_hash = opt.dry_run or \
            opt.action == Act.addresses or \
            opt.no_hash
        self.no_text = opt.dry_run or \
            opt.action == Act.addresses or \
            opt.no_text
        self.no_addr = opt.action != Act.addresses
        self.trace = opt.trace_parser
        self.diffs = []
        self.files = {}
        self.hashes = {}
        self.state = self.NONE_STATE
        self.text = None
        self.lno = 0

    def name(self):
        return self.input or '<stdin>'

    def debug(self, msg, *args):
        cerr("!!! %s:%d: %s\n", self.name(), self.lno, msg % args)

    def warn(self, msg, *args):
        warn("%s:%d: %s", self.name(), self.lno, msg % args)

    def error(self, pos, msg, *args):
        error("%s:%d: %s", self.name(), pos, msg % args)

    def error_context(self):
        from traceback import extract_stack
        for t in reversed(extract_stack()):
            if t[2].startswith('parse_'):
                return t[2][6:].replace('_', ' ')
        return None

    def invalid_line(self, *args):
        c = self.error_context()
        assert c is not None
        e = "invalid %s" % c
        if len(args):
            e += (": " + args[0]) % args[1:]
        self.error(self.lno, "%s", e)

    def source_not_found(self, name):
        self.invalid_line(
            "source file '%s' not found", name)

    def non_wsp_text_after(self, token):
        self.invalid_line(
            "non-whitespace text "
            "after '%s' in %s context",
            token, self.STATES[self.state])

    def token_not_allowed(self, token):
        self.error(self.lno,
            "token '%s' not allowed in %s context",
            token, self.STATES[self.state])

    def line_not_allowed(self, name):
        self.error(self.lno,
            "%s lines not allowed in %s context",
            name, self.STATES[self.state])

    NONE_STATE = 0
    DIFF_STATE = 1
    SOURCE_STATE = 2
    TARGET_STATE = 3

    STATES = (
        "NONE",   # NONE_STATE
        "DIFF",   # DIFF_STATE
        "SOURCE", # SOURCE_STATE
        "TARGET", # TARGET_STATE
    )

    def print_trace(self, state, token):
        n = self.STATES[state]
        self.debug("state=%r%*stoken=%r",
            n, 7 - len(n), "", token)

    def new_file(self, name):
        d = self.no_hash and \
            self.no_text and \
            self.no_addr
        f = File(name, d)
        if not d and not f.exists():
            self.source_not_found(name)
        g = self.files.get(f)
        if g is not None:
            return g
        self.files[f] = f
        return f

    def make_addr(self, left, middle, right):
        if not self.no_addr:
            return left + "%s" + middle + "%d" + right
        else:
            return None

    def make_lineno(self, lno):
        r = int(lno)
        assert r >= 0
        if r == 0:
            self.invalid_line(
                "zero line number")
        return r

    class Rex:

        def __init__(self, expr):
            self.expr = expr
            self.regex = None
            self.obj = None

        def match(self, text):
            if self.regex is None:
                self.regex = re.compile(self.expr)
                assert self.regex is not None
            self.obj = self.regex.match(text)
            return self.obj is not None

        def __getitem__(self, index):
            assert self.obj is not None
            return self.obj.group(index)

    def parse_comment(self, line):
        assert self.state != self.NONE_STATE

        if self.state != self.DIFF_STATE:
            self.line_not_allowed("comment")

        return self.state

    FILE_HASH = Rex('^###\s*([^\s:]+)'
        '(?:\s*:\s*(SHA\d+))?\s*:\s*([0-9a-fA-F]+)\s*$')

    def parse_file_hash(self, line):
        assert self.state != self.NONE_STATE

        if self.state != self.DIFF_STATE:
            self.token_not_allowed("###")
        if not self.FILE_HASH.match(line):
            self.invalid_line()

        h = PlainHash(
            file = self.new_file(
                self.FILE_HASH[1]),
            algo = self.FILE_HASH[2],
            digest = self.FILE_HASH[3],
            pos = self.lno
        )

        g = self.hashes.get(h.file)
        if g is not None:
            assert h.file == g.file
            self.invalid_line(
                "duplicate hash line: "
                "previous is at %d",
                g.pos)

        a = Hash(h.algo or 'SHA1')
        if len(h.digest) != a.digest_size():
            self.invalid_line(
                "invalid hash size: "
                "expected '%d' but got '%d'",
                a.digest_size(),
                len(h.digest))

        if not self.no_hash:
            d = a.digest(h.file)
            if d != h.digest:
                self.invalid_line(
                    "%s-digest mismatch: "
                    "expected '%s' but got '%s'",
                    a.name, h.digest, d)

        self.hashes[h.file] = h
        return self.state

    SPACES = Rex('^\s*$')

    is_except = staticmethod(
            PlainDiff.is_except)

    def parse_text_line(self, line):
        assert self.state != self.NONE_STATE

        if self.state == self.SOURCE_STATE or \
            self.state == self.TARGET_STATE:
            assert self.text is not None
            assert len(line)

            c = line[0]
            if c == '\\':
                t = line[1:]
                if t == '' or t == '\n':
                    self.invalid_line(
                        "unexpected end of line")
                if not self.is_except(t):
                    self.invalid_line(
                        "invalid escape sequence")
            else:
                t = line
                if c != '\n':
                    assert not self.is_except(t)
                else:
                    assert len(t) == 1

            self.text.append(t)

        elif not self.SPACES.match(line):
            self.line_not_allowed("non-empty")

        return self.state

    DIFF_ADDR = Rex('^([^\s:]+)(\s*:\s*)(\d+)$')

    def parse_diff_address(self, address, left, right):
        assert self.state == self.DIFF_STATE

        if not self.DIFF_ADDR.match(address):
            self.invalid_line()

        d = PlainDiff(
            file = self.new_file(
                    self.DIFF_ADDR[1]),
            line = self.make_lineno(
                    self.DIFF_ADDR[3]),
            addr = self.make_addr(
                left, self.DIFF_ADDR[2], right),
            pos = self.lno
        )
        self.diffs.append(d)

        return d

    DIFF_RIGHT = Rex('^(>>>\s*)(.*?)(\s*)$')

    def parse_diff_right(self, line):
        assert self.state != self.NONE_STATE
        assert self.DIFF_RIGHT.match(line)

        if self.state == self.DIFF_STATE:
            d = self.parse_diff_address(
                self.DIFF_RIGHT[2],
                self.DIFF_RIGHT[1],
                self.DIFF_RIGHT[3])

            self.text = d.source
            return self.SOURCE_STATE

        else:
            # self.state == self.SOURCE_STATE or
            # self.state == self.TARGET_STATE
            if len(self.DIFF_RIGHT[2]):
                self.non_wsp_text_after(">>>")

            d = self.diffs[-1]
            if len(d.source) == 0 and \
                len(d.target) == 0:
                self.error(d.pos, 
                    "diff hunk has both source "
                    "and target texts empty")

            self.text = None
            return self.DIFF_STATE

    DIFF_LEFT = Rex('^(<<<\s*)(.*?)(\s*)$')

    def parse_diff_left(self, line):
        assert self.state != self.NONE_STATE

        if self.state == self.DIFF_STATE or \
            self.state == self.SOURCE_STATE:
            assert self.DIFF_LEFT.match(line)

        if self.state == self.DIFF_STATE:
            d = self.parse_diff_address(
                self.DIFF_LEFT[2],
                self.DIFF_LEFT[1],
                self.DIFF_LEFT[3])

        elif self.state == self.SOURCE_STATE:
            if len(self.DIFF_LEFT[2]):
                self.non_wsp_text_after("<<<")

            d = self.diffs[-1]
            if not self.no_text:
                l = set()
                if not d.check_text(l):
                    assert len(l) > 0
                    l = sorted(l, key = int)
                    self.error(d.pos + l[0],
                        "%s:%d: diff text not"
                        " matching source text",
                        d.file, d.line + l[0] - 1)

        else:
            self.token_not_allowed("<<<")

        self.text = d.target
        return self.TARGET_STATE

    class Token:

        def __init__(self, lexeme, parse):
            self.lexeme = lexeme
            self.parse = parse

        def __str__(self):
            n = self.parse.__name__
            assert n[:6] == 'parse_'
            return n[6:].upper()

        def __repr__(self):
            return repr(str(self))

    TOKENS = (
        Token(
            lexeme = '///',
            parse = parse_comment,
        ),
        Token(
            lexeme = '###',
            parse = parse_file_hash,
        ),
        Token(
            lexeme = '>>>',
            parse = parse_diff_right,
        ),
        Token(
            lexeme = '<<<',
            parse = parse_diff_left,
        ),
        Token(
            lexeme = None,
            parse = parse_text_line,
        ),
    )

    def parse_file(self, file):
        self.state = self.DIFF_STATE
        self.lno = 0

        while True:
            l = file.readline()
            assert isinstance(l, str)

            if len(l) == 0:
                break
            self.lno += 1

            p = l[0:3]
            s = self.NONE_STATE
            for t in self.TOKENS:
                if t.lexeme == p or \
                    t.lexeme is None:
                    s = t.parse(self, l)
                    break

            if self.trace:
                self.print_trace(s, t)

            assert s != self.NONE_STATE
            self.state = s

        if self.state != self.DIFF_STATE:
            self.invalid_line(
                "unexpected end of file")

        return Plain(
            self.hashes, self.diffs)

    def parse(self):
        if self.input is None:
            f = sys.stdin
        else:
            try:
                f = open(self.input)
            except:
                error("opening '%s'", self.input)
        try:
            return self.parse_file(f)
        finally:
            f.close()

class Act:

    @staticmethod
    def parse(opt):
        p = Parser(opt)
        return p.parse()

    @staticmethod
    def plain(opt):
        p = Act.parse(opt)
        p.write(cout)

    @staticmethod
    def unified(opt):
        p = Act.parse(opt)
        u = Unified(
            diffs = p.diffs,
            opt = opt)
        u.write(cout)

    @staticmethod
    def addresses(opt):
        p = Act.parse(opt)
        a = Addresses(
            diffs = p.diffs,
            opt = opt)
        a.write(cout)

class Options:

    def __init__(self):
        self.action = Act.unified
        self.input = None
        self.context = 3
        self.directory = None
        self.dry_run = False
        self.no_hash = False
        self.no_text = False
        self.no_adjust = False
        self.no_reorder = False
        self.no_hunk_out = False
        self.interactive = False
        self.trace_parser = False
        self.verbose = False
        self.parse()

    def parse(self):
        args = sys.argv[1:]

        help = \
"""\
usage: %s [ACTION|OPTION]... [FILE]
where the actions are:
  -O|--parse-only          only parse input -- no output generated
  -P|--plain-diff          generate the normalized plain diff
  -U|--unified-diff        generate an unified diff (default)
  -A|--addresses-diff      generate the addresses diff; note that this
                             action option always implies using source file
                             content -- thus making option `--no-text-check'
                             to have no effect; also note that this action
                             option implies `--no-hash-check'
and the options are:
  -c|--context=NUM         output NUM lines of unified context (default: 3)
  -d|--directory=DIR       change the current directory to DIR immediately,
                             before doing anything else
  -i|--input=FILE          input plain diff file (default: '-', i.e. stdin)
     --[no-]hash-check     verify hash digests of source files or otherwise
                             do not (default do)
     --[no-]text-check     verify diff text lines matching source files or
                             otherwise do not (default do)
     --[no-]adjust-diff    use source file content for adjusting generated
                             unified context lines or not (default do)
     --[no-]dry-run        proceed the invoked action without actually
                             touching source files; by default do look up
                             for source files and use their content when
                             needed; note that `--dry-run' overrides any
                             of the options `--hash-check', `--text-check'
                             and `--adjust-diff'; note that `--[no-]dry-run'
                             doesn't relate to action `-A|--addresses-diff':
                             this action always implies using source files
     --[no-]reorder-hunks  reorder diff hunks referring to identical source
                             files by increasing source line numbers -- or
                             otherwise do not (default do)
     --[no-]hunk-output    modify the normal behavior of the program when
                             its action is `-A|--addresses-diff': instead
                             of producing diff hunks, make the program to
                             list all hunks in the given input file that
                             do not match the specified text lines of the
                             corresponding source file
     --[no-]interactive    prompt the user for each non-matching diff hunk
                             having multiple address choices, when action
                             is `-A|--addresses-diff'; if non-interactive,
                             the program exits with error the first time
                             it finds non-matching diff hunks (the default
                             behavior: non-interactive)
     --[no-]trace-parser   print out a trace of parsing the input file or
                             otherwise do not (default not)
     --dump-opts           print options and exit
  -v|--verbose             be verbose
     --version             print version numbers and exit
  -?|--help                display this help info and exit
"""

        class bits:

            help = False
            version = False
            dump = False

        def usage():
            cout(help, program)

        def version():
            cout("%s: version %s\n\n%s\n",
                    program, verdate, license)

        def dump():
            cout("""\
action:       %s
input:        %s
context:      %s
directory:    %s
dry-run:      %s
no-hash:      %s
no-text:      %s
no-adjust:    %s
no-reorder:   %s
no-hunk-out:  %s
interactive:  %s
trace-parser: %s
verbose:      %s
""",
            self.action.__name__,
            self.input or '<stdin>',
            self.context,
            self.directory,
            self.dry_run,
            self.no_hash,
            self.no_text,
            self.no_adjust,
            self.no_reorder,
            self.no_hunk_out,
            self.interactive,
            self.trace_parser,
            self.verbose)

        def invalid_arg(opt, arg, *extra):
            assert isinstance(arg, str)
            if len(extra):
                arg = arg % extra
            else:
                arg = "'%s'" % arg
            error("invalid argument for '%s' option: %s", \
                opt, arg)

        def parse_int(opt, arg):
            try:
                r = int(arg)
            except:
                invalid_arg(opt, arg)
            if r <= 0:
                invalid_arg(opt, arg)
            return r

        def parse_file_name(opt, arg):
            if arg != '-' and not os.access(arg, os.F_OK):
                if opt is not None:
                    invalid_arg(opt, "file '%s' not found", arg)
                else:
                    error("input file '%s' not found", arg)
            if arg == '-':
                return None
            else:
                return arg

        def parse_dir_name(opt, arg):
            import stat
            if arg != '-' and not (
                os.access(arg, os.F_OK) and
                stat.S_ISDIR(os.stat(arg).st_mode)):
                invalid_arg(opt, "dir '%s' not found", arg)
            if arg == '-':
                return None
            else:
                return arg

        from getopt import gnu_getopt, GetoptError
        try:
            opts, args = gnu_getopt(args,
                '?Ac:d:i:OPUv', (
                'context=',
                'directory=',
                'dry-run',
                'no-dry-run',
                'dump-opts',
                'input=',
                'hash-check',
                'no-hash-check',
                'text-check',
                'no-text-check',
                'adjust-diff',
                'no-adjust-diff',
                'reorder-hunks',
                'no-reorder-hunks',
                'hunk-output',
                'no-hunk-output',
                'interactive',
                'no-interactive',
                'parse-only',
                'plain-diff',
                'unified-diff',
                'addresses-diff',
                'trace-parser',
                'no-trace-parser',
                'verbose',
                'help',
                'version'))
        except GetoptError, msg:
            error(msg)

        for opt, arg in opts:
            if opt in ('-c', '--context'):
                self.context = parse_int(opt, arg)
            elif opt in ('-d', '--directory'):
                self.directory = parse_dir_name(opt, arg)
            elif opt in ('-i', '--input'):
                self.input = parse_file_name(opt, arg)
            elif opt in ('-O', '--parse-only'):
                self.action = Act.parse
            elif opt in ('-P', '--plain-diff'):
                self.action = Act.plain
            elif opt in ('-U', '--unified-diff'):
                self.action = Act.unified
            elif opt in ('-A', '--addresses-diff'):
                self.action = Act.addresses
            elif opt == '--dry-run':
                self.dry_run = True
            elif opt == '--no-dry-run':
                self.dry_run = False
            elif opt == '--hash-check':
                self.no_hash = False
            elif opt == '--no-hash-check':
                self.no_hash = True
            elif opt == '--text-check':
                self.no_text = False
            elif opt == '--no-text-check':
                self.no_text = True
            elif opt == '--adjust-diff':
                self.no_adjust = False
            elif opt == '--no-adjust-diff':
                self.no_adjust = True
            elif opt == '--reorder-hunks':
                self.no_reorder = False
            elif opt == '--no-reorder-hunks':
                self.no_reorder = True
            elif opt == '--hunk-output':
                self.no_hunk_out = False
            elif opt == '--no-hunk-output':
                self.no_hunk_out = True
            elif opt == '--interactive':
                self.interactive = True
            elif opt == '--no-interactive':
                self.interactive = False
            elif opt == '--trace-parser':
                self.trace_parser = True
            elif opt == '--no-trace-parser':
                self.trace_parser = False
            elif opt in ('-v', '--verbose'):
                self.verbose = True
            elif opt == '--dump-opts':
                bits.dump = True
            elif opt == '--version':
                bits.version = True
            elif opt in ('-?', '--help'):
                bits.help = True
            else:
                assert False

        if len(args):
            self.input = parse_file_name(None, args[0])

        if bits.version:
            version()
        if bits.dump:
            dump()
        if bits.help:
            usage()
        if bits.version or \
           bits.dump or \
           bits.help:
            sys.exit(0)

        if self.directory is not None:
            if self.input is not None:
                self.input = os.path.realpath(self.input)
            os.chdir(self.directory)

        if self.interactive and \
            self.no_hunk_out and \
            self.action == Act.addresses:
            warn("`--interactive' has no effect on "
                "action `-A|--addresses-diff' given "
                "along with `--no-hunk-output'")

def main():
    opt = Options()
    opt.action(opt)

if __name__ == '__main__':
    try:
        main()
    except IOError, (e, m):
        from errno import EPIPE
        if e != EPIPE: raise
    except KeyboardInterrupt:
        try:
            sys.stdin.close()
        except:
            pass


