/**
 * Language coverage for the syntax highlighter.
 *
 * Runs in Node rather than a browser: this is pure tokenizer logic, and the
 * only assertions that need a DOM (token colours, contrast) live in the
 * browser-side highlight test.
 *
 * Run: node scripts/highlight-selftest.mjs
 */
import assert from 'node:assert/strict';

import { highlight, renderCode, resolveLang } from '../public/highlight.js';

let passed = 0;
const failures = [];

function check(name, fn) {
  try {
    fn();
    passed += 1;
    console.log(`  PASS  ${name}`);
  } catch (err) {
    failures.push(name);
    console.log(`  FAIL  ${name}\n        ${err.message}`);
  }
}

/**
 * Strips the markup the highlighter added and unescapes its entities, so the
 * result can be compared to the original source. Highlighting escapes on
 * purpose, so comparing its output to raw source without undoing that would
 * always "fail".
 */
const textOf = (html) =>
  html
    .replace(/<[^>]*>/g, '')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/&amp;/g, '&');

/** Classes present in a highlighted fragment. */
const classesOf = (html) => new Set([...html.matchAll(/class="([^"]+)"/g)].map((m) => m[1]));
const hasAll = (html, want) => want.every((c) => html.includes(`class="${c}"`));

const SAMPLES = [
  ['js', 'const x = "hi"; // note\nfunction go(a) { return a + 1; }', ['c-kw', 'c-str', 'c-com', 'c-fn', 'c-num']],
  ['ts', 'interface A { x: number }\nconst a: string = "s";\nlet b = true;', ['c-kw', 'c-type', 'c-str', 'c-lit']],
  ['python', 'def go(x):\n    return "a"  # done', ['c-kw', 'c-str', 'c-com', 'c-fn']],
  ['sh', '# note\necho $HOME "${x}" --flag | grep -i a', ['c-com', 'c-var', 'c-str', 'c-key']],
  ['json', '{"a": 1, "b": "two", "c": true}', ['c-key', 'c-str', 'c-num', 'c-lit']],
  ['css', '/* note */\n@media print { .a { color: #fff; content: "x"; } }', ['c-com', 'c-num', 'c-str', 'c-kw']],
  ['html', '<div class="a">text</div>', ['c-tag', 'c-str', 'c-kw']],
  ['sql', "SELECT id FROM t WHERE n = 1 AND ok = true -- note", ['c-kw', 'c-num', 'c-com', 'c-lit']],
  ['go', 'func go(x int) string { return "a" }', ['c-kw', 'c-type', 'c-str']],
  ['rust', 'fn go() -> i32 { let x = 1; x }', ['c-kw', 'c-type', 'c-num']],
  ['ruby', 'def go(x)\n  puts "hi" # note\nend', ['c-kw', 'c-str', 'c-com']],
  ['php', '<?php $x = "a"; // note', ['c-kw', 'c-var', 'c-str', 'c-com']],
  ['yaml', 'server:\n  host: localhost\n  port: 8080  # the port\n  debug: true\n', ['c-key', 'c-num', 'c-com', 'c-lit']],
  ['yml', 'a: 1\nb: "two"\n', ['c-key', 'c-num', 'c-str']],
  ['toml', '[server]\nhost = "localhost"\nport = 8080\n', ['c-key', 'c-str', 'c-num']],
  ['ini', '; a comment\n[section]\nkey = value\n', ['c-key', 'c-com']],
  ['properties', '# note\napp.name=demo\napp.port=8080\napp.debug=true\n', ['c-key', 'c-com', 'c-num', 'c-lit']],
  ['dockerfile', 'FROM node:20\nRUN npm ci --production\nCOPY . /app\nARG VER=1\n', ['c-kw', 'c-str', 'c-key', 'c-num']],
  ['makefile', '# build\nall: app\n\t@echo "hi"\n', ['c-com', 'c-key', 'c-str']],
  ['graphql', 'query GetUser($id: ID!) {\n  user(id: $id) { name }\n}', ['c-kw', 'c-var', 'c-type']],
  ['terraform', '# note\nresource "aws_s3_bucket" "b" {\n  bucket = "x"\n}', ['c-com', 'c-key', 'c-str']],
  ['powershell', '# note\nGet-ChildItem -Path C:\\x -Force\nWrite-Host "hi"\n$n = 42\n', ['c-com', 'c-key', 'c-str', 'c-fn', 'c-var', 'c-num']],
  ['lua', '-- comment\nlocal function go(x)\n  return x + 1\nend\n', ['c-com', 'c-kw', 'c-num']],
  ['r', '# comment\nf <- function(x) { x + 1 }\ng(TRUE)\n', ['c-com', 'c-kw', 'c-num', 'c-lit']],
  ['perl', '# comment\nmy $x = \'hi\';\nprint $x;\n', ['c-com', 'c-var', 'c-str']],
  ['elixir', '# note\ndefmodule A do\n  @x 1\n  def go, do: :ok\nend\n', ['c-com', 'c-kw', 'c-var', 'c-lit']],
  ['haskell', '{- note -}\nmodule Main where\ngo :: Int -> Int\ngo x = x + 1\nmaybe Just\n', ['c-com', 'c-kw', 'c-type', 'c-lit']],
  ['clojure', '; note\n(defn go [x] (+ x 1))\n(map inc nil)\n', ['c-com', 'c-kw', 'c-lit']],
  ['swift', '// note\nfunc go(_ x: Int) -> Int { return x + 1 }\nlet s: String = "a"\n', ['c-com', 'c-kw', 'c-num', 'c-type', 'c-str']],
  ['dart', '// note\nint go(int x) => x + 1;\nString s = "a";\n', ['c-com', 'c-kw', 'c-num', 'c-type', 'c-str']],
  ['zig', '// note\npub fn go(x: i32) i32 { return x + 1; }\n', ['c-com', 'c-kw', 'c-type']],
  ['julia', '# note\nfunction go(x)\n  x + 1\nend\n', ['c-com', 'c-kw', 'c-num']],
  ['fsharp', 'let go x = x + 1\n', ['c-kw', 'c-num']],
  ['erlang', "% note\ngo() -> ok.\nf(1).\n", ['c-com', 'c-kw', 'c-lit', 'c-num']],
  ['solidity', '// note\ncontract A { function go() public pure returns (uint) { return 1; } }', ['c-com', 'c-kw', 'c-type', 'c-num']],
  ['objc', '// note\n@interface A : NSObject\n@end\n', ['c-com', 'c-tag', 'c-type']],
  ['latex', '% note\n\\begin{document}\nHello\n\\end{document}\n', ['c-com', 'c-kw']],
  ['asm', '; note\nmov eax, 1\nret\n', ['c-com', 'c-kw', 'c-num']],
  ['protobuf', '// note\nmessage User {\n  string name = 1;\n}\n', ['c-com', 'c-kw', 'c-type']],
  ['crystal', '# note\ndef go(x)\n  x + 1\nend\n', ['c-com', 'c-kw', 'c-num']],
  ['nim', '# note\nproc go(x: int): int =\n  x + 1\n', ['c-com', 'c-kw', 'c-type']],
  ['ocaml', '(* note *)\nlet go x = x + 1\n', ['c-com', 'c-kw', 'c-num']],
  ['nginx', '# note\nserver {\n  listen 80;\n}', ['c-com', 'c-kw', 'c-num']],
  ['cmake', '# note\nproject(app)\nset(SRC a.cpp)\nadd_library(x STATIC)\n', ['c-com', 'c-kw', 'c-var', 'c-str']],
  ['diff', '@@ -1 +1 @@\n-removed\n+added\n', ['c-key', 'c-add', 'c-del']],
  ['markdown', '# Title\n**bold**\n', ['c-key', 'c-str']],
];

console.log('\n=== every language resolves and emits its tokens ===');
for (const [label, src, want] of SAMPLES) {
  check(`${label}`, () => {
    const r = highlight(src, label);
    assert.ok(r, `${label} did not resolve`);
    const missing = want.filter((c) => !r.html.includes(`class="${c}"`));
    assert.equal(missing.length, 0, `missing ${missing.join(', ')}`);
    // Highlighting must only add markup, never change the code.
    assert.equal(textOf(r.html), src, 'highlighting altered the text');
  });
}

console.log('\n=== aliases resolve to a lexer ===');
for (const alias of ['yml', 'py', 'bash', 'console', 'ps1', 'docker', 'gql', 'tf',
                      'rs', 'kt', 'cs', 'objc', 'nasm', 'proto', 'ml', 'mk',
                      'yml', 'javascript', 'typescript', 'golang', 'html']) {
  check(`${alias} -> ${resolveLang(alias)}`, () => {
    assert.ok(resolveLang(alias), `${alias} did not resolve`);
  });
}

console.log('\n=== unknown labels degrade instead of guessing ===');
for (const bad of ['', 'notalanguage', 'brainfuck', null, undefined]) {
  check(`${JSON.stringify(bad)} -> null`, () => {
    assert.equal(resolveLang(bad), null);
  });
}

check('an unlabelled fence renders plain', () => {
  const html = renderCode('x = 1', '');
  assert.ok(!html.includes('class="hl"'), 'should not be marked highlighted');
  assert.ok(html.includes('x = 1'), 'text should survive');
});

console.log('\n=== escaping: code cannot inject markup ===');
check('a script tag is escaped', () => {
  const html = renderCode('<script>alert(1)</script>', 'js');
  assert.ok(!html.includes('<script>'), 'raw script tag present');
  assert.ok(html.includes('&lt;script&gt;'), 'escaped form missing');
});

check('an img with onerror is escaped', () => {
  const html = renderCode('<img src=x onerror=alert(1)>', '');
  assert.ok(!html.includes('<img'), 'raw img tag present');
});

check('ampersands are escaped exactly once', () => {
  const html = renderCode('a & b && c', 'js');
  assert.ok(html.includes('&amp;'), 'ampersand not escaped');
  assert.ok(!html.includes('&amp;amp;'), 'double-escaped');
});

check('the lang attribute cannot break out', () => {
  const html = renderCode('x', 'js"><script>alert(1)</script>');
  assert.ok(!html.includes('"><script>'), 'attribute escaped improperly');
});

console.log('\n=== keywords match whole words only ===');
check('`if` is a keyword but `gift` is not', () => {
  const html = highlight('gift and if', 'js').html;
  assert.ok(html.includes('<span class="c-kw">if</span>'), 'if should be a keyword');
  assert.ok(!html.includes('<span class="c-kw">gift</span>'), 'gift should not be');
});

check('sql keywords are case-insensitive', () => {
  for (const s of ['select', 'SELECT', 'Select']) {
    assert.ok(highlight(s, 'sql').html.includes('c-kw'), `${s} should be a keyword`);
  }
});

check('a quoted keyword inside a string stays a string', () => {
  const html = highlight('"if"', 'js').html;
  assert.ok(!html.includes('c-kw'), 'keyword coloured inside a string');
  assert.ok(html.includes('c-str'), 'string not recognised');
});

check('a quoted keyword inside a comment stays a comment', () => {
  const html = highlight('// "if"', 'js').html;
  assert.ok(html.includes('c-com'), 'comment not recognised');
  assert.ok(!html.includes('c-kw'), 'keyword coloured inside a comment');
});

console.log('\n=== unterminated constructs (mid-stream) stay coloured ===');
check('an unterminated string still reads as a string', () => {
  const html = highlight('const x = "unfinished', 'js').html;
  assert.ok(html.includes('c-str'), 'unterminated string not coloured');
});

check('an unterminated block comment still reads as a comment', () => {
  const html = highlight('/* unfinished', 'js').html;
  assert.ok(html.includes('c-com'), 'unterminated comment not coloured');
});

console.log('\n=== cost is linear, not quadratic ===');
const LINE = 'const value = compute(a, "x") + 1; // note\n';
const timings = {};
for (const n of [50, 200, 800]) {
  const code = LINE.repeat(n);
  const t0 = performance.now();
  for (let i = 0; i < 5; i++) renderCode(code, 'js');
  timings[n] = (performance.now() - t0) / 5;
}
check(`16x input stays near-linear (${timings[50].toFixed(2)}ms -> ${timings[800].toFixed(2)}ms)`, () => {
  const ratio = timings[800] / Math.max(timings[50], 0.001);
  assert.ok(ratio < 40, `16x input cost ${ratio.toFixed(0)}x time, expected near-linear`);
});

console.log(`\n${passed} passed, ${failures.length} failed`);
if (failures.length) {
  console.log(`\nFailing: ${failures.join(', ')}`);
  process.exit(1);
}
