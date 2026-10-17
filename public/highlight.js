/**
 * A small syntax highlighter for fenced code blocks.
 *
 * Hand-rolled rather than vendored: this app has no build step and serves
 * public/ raw, so a library would mean committing a minified vendor file and
 * fighting its DOM manipulation against the per-tick re-render that streaming
 * does. It also keeps every colour on the app's own theme tokens.
 *
 * This is a lexer, not a parser. It recognises the parts that carry meaning to a
 * reader — comments, strings, numbers, keywords, literals, types, call sites —
 * which is what makes code scannable. It does not understand scope or types, so
 * a preprocessor macro or a multi-line expression may be coloured loosely.
 *
 * Always returns HTML-escaped output: callers must NOT pre-escape the code, or
 * entities would end up doubled and visible.
 */

/**
 * Escapes for text content. Quotes are deliberately left alone: they only need
 * escaping inside an attribute value, and escaping them here would litter the
 * markup with &quot; for no benefit.
 */
const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

/** Escapes for a double-quoted attribute value, so a label cannot break out. */
const escAttr = (s) => esc(s).replace(/"/g, '&quot;');

/** Every rule is sticky: it may only match at lastIndex, never scan ahead. */
const rule = (re) => new RegExp(re.source, re.flags.includes('y') ? re.flags : `${re.flags}y`);

const r = (source, flags = '') => new RegExp(source, flags);
const R = (source, flags = '') => rule(new RegExp(source, flags));

/* ---------------- Language definitions ---------------- */

/**
 * Builds the rule list shared by config-style formats (yaml, toml, ini,
 * nginx, cmake, terraform). They all boil down to: a comment marker, quoted
 * strings, a key header, and bare scalars.
 */
function cfgLang(o) {
  const rules = [];
  if (o.hash) rules.push(['com', R(String.raw`#[^\n]*`)]);
  if (o.semi) rules.push(['com', R(String.raw`;[^\n]*`)]);
  if (o.slash) rules.push(['com', R(String.raw`//[^\n]*`)]);
  // [section] / [[section]] table headers, which read as keys.
  if (o.table) rules.push(['key', R(String.raw`^[ \t]*\[[^\]\n]*\]`, 'm')]);
  if (o.key) rules.push(['key', R(o.key, 'm')]);
  if (o.directive) rules.push(['kw', R(o.directive, 'm')]);
  if (o.equals) {
    rules.push(['key', R(String.raw`^[ \t]*[\w.$/-]+(?=[ \t]*=[^=])`, 'm')]);
  } else {
    rules.push(['key', R(String.raw`^[ \t]*(?:-[ \t]+)?[\w.$/-]+(?=[ \t]*:(?:[ \t]|$))`, 'm')]);
  }
  rules.push(['str', R(String.raw`"""[\s\S]*?"""`)]);
  rules.push(['str', R(String.raw`"(?:\\[\s\S]|[^"\\])*"?`)]);
  rules.push(['str', R(String.raw`'[^'\n]*'?`)]);
  for (const extra of o.extra ?? []) rules.push(extra);
  if (o.literals) {
    rules.push(['lit', R(`\\b(?:${o.literals.split(/\s+/).join('|')})\\b`, 'i')]);
  }
  rules.push(['num', R(String.raw`\b\d[\d_]*(?:\.\d+)?(?:[eE][+-]?\d+)?\b`)]);
  rules.push(['id', R(String.raw`[\w.$/-]+`)]);
  return {
    aliases: o.alias ?? [],
    rules,
    kw: o.keywords ?? '',
    lit: '',
    type: '',
  };
}

/**
 * Rules are tried in order at each position, and the first match wins — so
 * comments and strings are listed before identifiers. Otherwise the word `if`
 * inside a string would still be coloured as a keyword.
 *
 * `kw`/`lit`/`type` are matched against the whole identifier, so `if` never
 * matches inside `gift`.
 */
const LANGS = {
  javascript: {
    aliases: ['js', 'jsx', 'mjs', 'cjs', 'node'],
    rules: [
      ['com', R(String.raw`//[^\n]*`)],
      ['com', R(String.raw`/\*[\s\S]*?(?:\*\/|$)`)],
      // The trailing ? makes an unterminated string still colour as a string
      // mid-stream, instead of flashing as plain text until the quote arrives.
      ['str', R(String.raw`\`(?:\\[\s\S]|[^\`\\])*\`?`)],
      ['str', R(String.raw`"(?:\\[\s\S]|[^"\\\n])*"?`)],
      ['str', R(String.raw`'(?:\\[\s\S]|[^'\\\n])*'?`)],
      ['num', R(String.raw`\b0[xX][\da-fA-F_]+\b|\b\d[\d_]*(?:\.\d[\d_]*)?(?:[eE][+-]?\d+)?n?\b`)],
      ['id', R(String.raw`[A-Za-z_$][\w$]*`)],
    ],
    kw: `const let var function return if else for while do break continue new class extends super this typeof instanceof in of delete void throw try catch finally switch case default async await yield static get set import export from as`,
    lit: `true false null undefined NaN Infinity`,
    type: `Array Object String Number Boolean Promise Map Set WeakMap WeakSet Symbol Date RegExp Error JSON Math console window document`,
  },

  typescript: {
    aliases: ['ts', 'tsx'],
    rules: [
      ['com', R(String.raw`//[^\n]*`)],
      ['com', R(String.raw`/\*[\s\S]*?(?:\*\/|$)`)],
      ['str', R(String.raw`\`(?:\\[\s\S]|[^\`\\])*\`?`)],
      ['str', R(String.raw`"(?:\\[\s\S]|[^"\\\n])*"?`)],
      ['str', R(String.raw`'(?:\\[\s\S]|[^'\\\n])*'?`)],
      ['num', R(String.raw`\b0[xX][\da-fA-F_]+\b|\b\d[\d_]*(?:\.\d[\d_]*)?(?:[eE][+-]?\d+)?n?\b`)],
      ['id', R(String.raw`[A-Za-z_$][\w$]*`)],
    ],
    kw: `const let var function return if else for while do break continue new class extends implements interface type enum as this typeof instanceof in of delete void throw try catch finally switch case default async await yield static readonly public private protected abstract declare namespace module import export from satisfies keyof infer asserts is unknown never override accessor readonly`,
    lit: `true false null undefined NaN Infinity`,
    type: `string number boolean any unknown never void object symbol bigint Array Object String Number Boolean Promise Map Set Date RegExp Error JSON Math console`,
  },

  python: {
    aliases: ['py', 'python3', 'py3'],
    rules: [
      ['com', R(String.raw`#[^\n]*`)],
      // Triple-quoted must come first: the single-quote rules would otherwise
      // match `"""` as an empty string plus a stray quote.
      ['str', R(String.raw`[rbfu]{0,2}(?:"""[\s\S]*?"""|'''[\s\S]*?''')`)],
      ['str', R(String.raw`[rbfu]{0,2}"(?:\\[\s\S]|[^"\\\n])*"?`)],
      ['str', R(String.raw`[rbfu]{0,2}'(?:\\[\s\S]|[^'\\\n])*'?`)],
      ['num', R(String.raw`\b0[xXbBoO][\da-fA-F_]+\b|\b\d[\d_]*(?:\.\d[\d_]*)?(?:[eE][+-]?\d+)?j?\b`)],
      ['id', R(String.raw`[A-Za-z_][\w]*`)],
    ],
    kw: `def class return if elif else for while break continue pass import from as try except finally raise with lambda global nonlocal assert del yield async await not and or is in match case`,
    lit: `True False None self cls`,
    type: `int str float bool list dict set tuple bytes object type range len print open enumerate zip map filter sum min max abs sorted reversed isinstance super property staticmethod classmethod`,
  },

  shell: {
    aliases: ['sh', 'bash', 'zsh', 'shell', 'console', 'terminal'],
    rules: [
      ['com', R(String.raw`#[^\n]*`)],
      ['str', R(String.raw`"(?:\\[\s\S]|[^"\\])*"?`)],
      ['str', R(String.raw`'[^']*'?`)],
      ['var', R(String.raw`\$\{[^}\n]*\}|\$[A-Za-z_]\w*|\$[0-9@*#?$!-]`)],
      ['num', R(String.raw`\b\d+\b`)],
      // Long options first, so `--color` isn't split into `-` + `-color`.
      ['id', R(String.raw`-{1,2}[A-Za-z][\w-]*`)],
      ['id', R(String.raw`[A-Za-z_][\w]*`)],
    ],
    kw: `if then else elif fi for while until do done case esac function return break continue local export readonly declare source eval exec set unset trap shift`,
    lit: `true false`,
    type: `echo cd ls cat grep sed awk curl git npm node python pip docker sudo chmod mkdir rm cp mv touch`,
  },

  json: {
    aliases: ['jsonc', 'json5'],
    rules: [
      // A quoted string followed by a colon is a key, not a value.
      ['key', R(String.raw`"(?:\\[\s\S]|[^"\\])*"(?=\s*:)`)],
      ['str', R(String.raw`"(?:\\[\s\S]|[^"\\])*"?`)],
      ['num', R(String.raw`-?\b\d[\d_]*(?:\.\d+)?(?:[eE][+-]?\d+)?\b`)],
      ['lit', R(String.raw`\b(?:true|false|null)\b`)],
    ],
    kw: ``,
    lit: ``,
    type: ``,
  },

  css: {
    aliases: ['scss', 'less', 'postcss'],
    rules: [
      ['com', R(String.raw`/\*[\s\S]*?(?:\*\/|$)`)],
      ['str', R(String.raw`"(?:\\[\s\S]|[^"\\\n])*"?`)],
      ['str', R(String.raw`'[^'\n]*'?`)],
      ['kw', R(String.raw`@[A-Za-z-]+`)],
      ['num', R(String.raw`#[0-9a-fA-F]{3,8}\b|\b-?\d*\.?\d+(?:px|em|rem|%|vh|vw|s|ms|deg|fr|ch|ex)?\b`)],
      ['id', R(String.raw`--[\w-]+|[-A-Za-z_][\w-]*`)],
    ],
    kw: `important media import charset keyframes supports layer container property from to and not only`,
    lit: `inherit initial unset revert auto none`,
    type: `flex grid block inline absolute relative fixed sticky hidden transparent currentColor`,
  },

  html: {
    aliases: ['xml', 'svg', 'xhtml', 'vue', 'svelte'],
    rules: [
      ['com', R(String.raw`<!--[\s\S]*?-->`)],
      ['tag', R(String.raw`</?[A-Za-z][\w:.-]*`)],
      ['str', R(String.raw`"[^"]*"?`)],
      ['str', R(String.raw`'[^']*'?`)],
      ['kw', R(String.raw`/?>`)],
      ['id', R(String.raw`[-A-Za-z_:][\w:.-]*`)],
    ],
    kw: `DOCTYPE html head body div span script style link meta title a p ul ol li table tr td th form input button label select option textarea section header footer nav main aside article template`,
    lit: ``,
    type: ``,
  },

  sql: {
    aliases: ['postgres', 'postgresql', 'mysql', 'sqlite', 'plsql'],
    ci: true,
    rules: [
      ['com', R(String.raw`--[^\n]*`)],
      ['com', R(String.raw`/\*[\s\S]*?(?:\*\/|$)`)],
      ['str', R(String.raw`'(?:''|[^'])*'?`)],
      ['num', R(String.raw`\b\d+(?:\.\d+)?\b`)],
      // Quoted identifiers too, so `"user"` and `[table]` are not left bare.
      ['id', R(String.raw`\`[^\`]*\`?|"(?:""|[^"])*"?|\[?[A-Za-z_][\w$]*\]?`)],
    ],
    kw: `select from where insert into values update set delete create alter drop table view index join inner left right outer full cross on using group by order having limit offset union all distinct as returning with recursive`,
    lit: `true false null`,
    type: `int integer bigint smallint serial text varchar char boolean date timestamp timestamptz numeric decimal real double float blob uuid json jsonb array`,
  },

  go: {
    aliases: ['golang'],
    rules: [
      ['com', R(String.raw`//[^\n]*`)],
      ['com', R(String.raw`/\*[\s\S]*?(?:\*\/|$)`)],
      ['str', R(String.raw`\`[^\`]*\`?`)],
      ['str', R(String.raw`"(?:\\[\s\S]|[^"\\\n])*"?`)],
      ['str', R(String.raw`'(?:\\.|[^'\\])'`)],
      ['num', R(String.raw`\b0[xX][\da-fA-F_]+\b|\b\d[\d_]*(?:\.\d+)?(?:[eE][+-]?\d+)?\b`)],
      ['id', R(String.raw`[A-Za-z_]\w*`)],
    ],
    kw: `package import func return if else for range switch case default break continue go defer var const type struct interface map chan select fallthrough goto`,
    lit: `true false nil iota`,
    type: `string int int8 int16 int32 int64 uint uint8 uint16 uint32 uint64 uintptr byte rune float32 float64 complex64 complex128 bool error any make new len cap append copy delete panic recover print println fmt os strconv time context sync`,
  },

  rust: {
    aliases: ['rs'],
    rules: [
      ['com', R(String.raw`//[^\n]*`)],
      ['com', R(String.raw`/\*[\s\S]*?(?:\*\/|$)`)],
      ['str', R(String.raw`r?#*"(?:\\[\s\S]|[^"\\])*"?#*`)],
      ['str', R(String.raw`'(?:\\.|[^'\\])'`)],
      ['num', R(String.raw`\b0[xXbBoO][\da-fA-F_]+\b|\b\d[\d_]*(?:\.\d+)?(?:[eE][+-]?\d+)?(?:[iuf](?:8|16|32|64|128|size))?\b`)],
      ['id', R(String.raw`[A-Za-z_]\w*`)],
    ],
    kw: `fn let mut const static struct enum impl trait for while loop if else match return use mod pub crate self super where as ref move dyn async await unsafe type in break continue`,
    lit: `true false None Some Ok Err`,
    type: `String Vec Option Result Box i8 i16 i32 i64 i128 u8 u16 u32 u64 u128 usize isize f32 f64 bool char str println vec format panic`,
  },

  java: {
    aliases: ['kotlin', 'kt', 'kts', 'groovy', 'scala', 'csharp', 'cs', 'cpp', 'c'],
    rules: [
      ['com', R(String.raw`//[^\n]*`)],
      ['com', R(String.raw`/\*[\s\S]*?(?:\*\/|$)`)],
      ['str', R(String.raw`"""[\s\S]*?"""`)],
      ['str', R(String.raw`"(?:\\[\s\S]|[^"\\\n])*"?`)],
      ['str', R(String.raw`'(?:\\.|[^'\\])'`)],
      ['num', R(String.raw`\b0[xXbB][\da-fA-F_]+\b|\b\d[\d_]*(?:\.\d+)?(?:[eE][+-]?\d+)?[lLfFdD]?\b`)],
      ['id', R(String.raw`@[A-Za-z_]\w*`)],
      ['id', R(String.raw`[A-Za-z_$][\w$]*`)],
    ],
    kw: `public private protected class interface enum extends implements import package new return if else for while do break continue try catch finally throw throws static final abstract void val var fun object when companion internal data sealed override lateinit`,
    lit: `true false null this super`,
    type: `int long short byte char float double boolean String Integer Long Double Boolean List ArrayList Map HashMap Set Optional Stream`,
  },

  glsl: {
    aliases: [],
    rules: [
      ['com', R(String.raw`//[^
]*`)],
      ['com', R(String.raw`/\*[\s\S]*?(?:\*/|$)`)],
      ['str', R(String.raw`"(?:\\[\s\S]|[^"\\])*"?`)],
      ['str', R(String.raw`'[^'\n]*'?`)],
      ['num', R(String.raw`\b0[xX][\da-fA-F_]+\b|\b\d[\d_]*(?:\.\d+)?(?:[eE][+-]?\d+)?\b`)],
      ['id', R(String.raw`[A-Za-z_][\w]*`)],
    ],
    kw: `attribute const varying const bool float int void true false null precision in out inout`,
    lit: ``,
    type: `float int bool void vec2 vec3 vec4 mat2 mat3 mat4`,
  },

  ruby: {
    aliases: ['rb'],
    rules: [
      ['com', R(String.raw`#[^\n]*`)],
      ['str', R(String.raw`"(?:\\[\s\S]|[^"\\])*"?`)],
      ['str', R(String.raw`'[^'\n]*'?`)],
      ['num', R(String.raw`\b0[xXbB][\da-fA-F_]+\b|\b\d[\d_]*(?:\.\d+)?\b`)],
      ['id', R(String.raw`:[A-Za-z_]\w*[?!]?`)],
      ['id', R(String.raw`@@?[A-Za-z_]\w*`)],
      ['id', R(String.raw`[A-Za-z_]\w*[?!]?`)],
    ],
    kw: `def end class module if elsif else unless while until do begin rescue ensure raise return yield require require_relative attr_accessor attr_reader attr_writer include extend private public protected nil then case when and or not`,
    lit: `nil true false`,
    type: `puts print p new lambda proc Array Hash String Integer Float Symbol`,
  },

  php: {
    aliases: [],
    rules: [
      ['com', R(String.raw`//[^\n]*`)],
      // # is a comment in PHP but #[...] is an attribute.
      ['com', R(String.raw`#(?!\[)[^\n]*`)],
      ['com', R(String.raw`/\*[\s\S]*?(?:\*\/|$)`)],
      ['var', R(String.raw`\$[A-Za-z_]\w*`)],
      ['str', R(String.raw`"(?:\\[\s\S]|[^"\\])*"?`)],
      ['str', R(String.raw`'[^'\n]*'?`)],
      ['num', R(String.raw`\b\d[\d_]*(?:\.\d+)?\b`)],
      ['id', R(String.raw`[A-Za-z_$\\]?][\w$]*`)],
    ],
    kw: `function return if else elseif foreach for while do end class new public private protected static abstract final const try catch throw use namespace extends implements require include echo print var`,
    lit: `true false null TRUE FALSE NULL`,
    type: `string int float bool array object void self this`,
  },

  /**
   * Config formats are all the same shape — comments, quoted strings, a
   * `key:` or `key=` header, scalars — so they share one rule builder and vary
   * only in their comment marker and delimiters.
   */
  yaml: cfgLang({
    alias: ['yml'],
    hash: true,
    // A key is a line-leading token immediately followed by `:`.
    key: String.raw`^[ \t]*(?:-[ \t]+)?[\w.$/-]+(?=[ \t]*:(?:[ \t]|$))`,
    literals: 'true false null yes no on off ~',
    extra: [['var', R(String.raw`&[\w-]+|\*[\w-]+`)]],
  }),

  toml: cfgLang({
    alias: [],
    hash: true,
    table: true,
    equals: true,
    literals: 'true false',
  }),

  ini: cfgLang({
    alias: ['cfg', 'conf', 'properties', 'editorconfig', 'desktop', 'env'],
    hash: true,
    semi: true,
    table: true,
    equals: true,
    literals: 'true false yes no on off',
  }),

  nginx: cfgLang({
    alias: ['conf'],
    hash: true,
    // Directives are a bare word at the head of a line.
    directive: String.raw`^[ \t]*[a-z_]+(?=[ \t;(])`,
    extra: [['var', R(String.raw`\$\{?\w+\}?`)]],
  }),

  cmake: cfgLang({
    alias: [],
    hash: true,
    directive: String.raw`^[ \t]*[A-Za-z_][\w]*[ \t]*(?=\()`,
    extra: [['var', R(String.raw`\$\{[^}\n]*\}|\$[A-Za-z_]\w*`)]],
  }),

  terraform: cfgLang({
    alias: ['tf', 'tfvars', 'hcl'],
    hash: true,
    slash: true,
    equals: true,
    literals: 'true false null',
    keywords: `resource variable output provider module locals data terraform for_each count depends_on lifecycle sensitive`,
  }),

  dockerfile: {
    aliases: ['docker', 'containerfile'],
    rules: [
      ['com', R(String.raw`#[^\n]*`)],
      ['str', R(String.raw`"(?:\\[\s\S]|[^"\\])*"?`)],
      ['str', R(String.raw`'[^'\n]*'?`)],
      ['var', R(String.raw`\$\{?\w+\}?`)],
      ['flag', R(String.raw`--[\w-]+`)],
      ['num', R(String.raw`\b\d[\d_]*\b`)],
      ['id', R(String.raw`[A-Za-z_][\w.-]*`)],
    ],
    kw: `FROM RUN CMD LABEL MAINTAINER EXPOSE ENV ADD COPY ENTRYPOINT VOLUME USER WORKDIR ARG ONBUILD STOPSIGNAL HEALTHCHECK SHELL AS`,
    lit: ``,
    type: `latest alpine node python ubuntu debian`,
  },

  makefile: {
    aliases: ['mk', 'make'],
    rules: [
      ['com', R(String.raw`#[^\n]*`)],
      ['var', R(String.raw`\$\$?[A-Za-z_]\w*|\$\([^)\n]*\)|\$\{[^}\n]*\}`)],
      ['key', R(String.raw`^[A-Za-z0-9_./%-]+(?=[ \t]*:(?!=))`, 'm')],
      ['str', R(String.raw`"(?:\\[\s\S]|[^"\\])*"?`)],
      ['str', R(String.raw`'[^'\n]*'?`)],
      ['num', R(String.raw`\b\d+\b`)],
      ['id', R(String.raw`[A-Za-z_][\w.-]*`)],
    ],
    kw: `include ifeq ifneq ifdef ifndef else endif define endef export unexport override vpath .PHONY`,
    lit: ``,
    type: `all clean install test build`,
  },

  graphql: {
    aliases: ['gql'],
    rules: [
      ['com', R(String.raw`#[^\n]*`)],
      ['str', R(String.raw`"""[\s\S]*?"""`)],
      ['str', R(String.raw`"(?:\\[\s\S]|[^"\\])*"?`)],
      ['var', R(String.raw`[$@]\w+`)],
      ['num', R(String.raw`\b\d+(?:\.\d+)?\b`)],
      ['id', R(String.raw`[_A-Za-z]\w*`)],
    ],
    kw: `query mutation subscription fragment on type input interface enum union scalar schema extend implements directive repeatable IN`,
    lit: `true false null`,
    type: `ID String Int Float Boolean`,
  },

  powershell: {
    aliases: ['ps1', 'psm1', 'pwsh'],
    rules: [
      ['com', R(String.raw`#[^\n]*`)],
      // No backtick in the exclusion class: PowerShell uses it as an escape
      // character, but a literal backtick would terminate the template literal
      // this is written in. Treating `"` as the terminator is close enough.
      ['str', R(String.raw`"(?:\\[\s\S]|[^"\n])*"?`)],
      ['str', R(String.raw`'[^'\n]*'?`)],
      ['var', R(String.raw`\$(?:\{[^}\n]*\}|[A-Za-z_]\w*)`)],
      ['flag', R(String.raw`-{1,2}[A-Za-z][\w-]*`)],
      ['num', R(String.raw`\b\d[\d.]*\b`)],
      ['id', R(String.raw`[A-Za-z_]\w*-?`)],
    ],
    kw: `if elseif else switch foreach for while do until break continue return function filter begin process end param throw try catch finally class enum using import in exit`,
    lit: `true false null`,
    type: `Write-Host Write-Output Write-Error Get- Set- New- Remove- Invoke- Start- Stop- Test- Select-Object Where-Object ForEach-Object Sort-Object`,
  },

  lua: {
    aliases: [],
    rules: [
      ['com', R(String.raw`--\[\[[\s\S]*?\]\]`)],
      ['com', R(String.raw`--[^\n]*`)],
      // Lua long strings: [=[ ... ]=]. The bracket level is variable, so match
      // `=` runs on both sides rather than assuming one.
      ['str', R(String.raw`\[=*\[[\s\S]*?\]=*\]`)],
      ['str', R(String.raw`"(?:\\[\s\S]|[^"\\])*"?`)],
      ['str', R(String.raw`'[^'\n]*'?`)],
      ['num', R(String.raw`\b0[xX][\da-fA-F]+(?:\.\w+)?\b|\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b`)],
      ['id', R(String.raw`[A-Za-z_][\w]*`)],
    ],
    kw: `and break do else elseif end for function goto if in local not or repeat return then until while`,
    lit: `true false nil`,
    type: `print require pairs ipairs table string tonumber tostring math io os coroutine setmetatable getmetatable select unpack pcall xpcall error assert`,
  },

  r: {
    aliases: ['rscript'],
    rules: [
      ['com', R(String.raw`#[^\n]*`)],
      ['str', R(String.raw`"(?:\\[\s\S]|[^"\\])*"?`)],
      ['str', R(String.raw`'[^'\n]*'?`)],
      ['num', R(String.raw`\b\d+\.?\d*(?:[eE][+-]?\d+)?[Li]?\b`)],
      ['id', R(String.raw`[A-Za-z.][\w.]*`)],
    ],
    kw: `if else repeat while function for in next break return library require source`,
    lit: `TRUE FALSE NULL NA Inf NaN T F`,
    type: `c list vector matrix data.frame length nrow ncol sum mean sd apply sapply lapply vapply paste paste0 print cat vector seq rep factor levels table`,
  },

  perl: {
    aliases: ['pl', 'pm'],
    rules: [
      ['com', R(String.raw`#[^\n]*`)],
      ['str', R(String.raw`qw[([{][^)\]}]*[)\]}]`)],
      ['str', R(String.raw`"(?:\\[\s\S]|[^"\\])*"?`)],
      ['str', R(String.raw`'(?:\\[\s\S]|[^'\\])*'?`)],
      ['var', R(String.raw`[$@%][{]?[$#]?[A-Za-z_]\w*[}]?`)],
      ['num', R(String.raw`\b\d[\d_]*(?:\.\d+)?(?:[eE][+-]?\d+)?\b`)],
      ['id', R(String.raw`[A-Za-z_]\w*`)],
    ],
    kw: `sub my our local if elsif else unless while until for foreach do last next redo return package use require no eval defined wantarray bless ref printf print sprintf`,
    lit: `undef __PACKAGE__ __END__ __DATA__`,
    type: `push pop shift unshift splice join split grep map sort reverse scalar keys values each exists delete die warn open close chomp chop substr index sprintf length lc uc`,
  },

  elixir: {
    aliases: ['ex', 'exs'],
    rules: [
      ['com', R(String.raw`#[^\n]*`)],
      ['str', R(String.raw`"""[\s\S]*?"""`)],
      ['str', R(String.raw`"(?:\\[\s\S]|[^"\\])*"?`)],
      ['var', R(String.raw`@[a-z_]\w*`)],
      ['lit', R(String.raw`:[A-Za-z_]\w*[!?]?`)],
      ['num', R(String.raw`\b\d[\d_]*(?:\.\d+)?\b`)],
      ['id', R(String.raw`[A-Za-z_]\w*[!?]?`)],
    ],
    kw: `def defp defmodule defstruct defprotocol defimpl defmacrop do end if else unless case cond with for fn when and or not in try rescue catch after receive raise throw alias import require use quote unquote`,
    lit: `true false nil __MODULE__`,
    type: `Enum Map List String Integer Float Atom Tuple Process IO Kernel Agent GenServer Supervisor Task`,
  },

  haskell: {
    aliases: ['hs'],
    rules: [
      ['com', R(String.raw`\{-[\s\S]*?-\}`)],
      ['str', R(String.raw`"(?:\\[\s\S]|[^"\\])*"?`)],
      ['var', R(String.raw`\b[a-z_]\w*'*\b`)],
      ['type', R(String.raw`\b[A-Z]\w*'*\b`)],
      ['num', R(String.raw`\b\d[\d.]*(?:[eE][+-]?\d+)?\b`)],
      ['id', R(String.raw`[!#$%&*+./<=>?@\\^|~-]+`)],
    ],
    kw: `module import qualified as hiding where data type newtype class instance deriving do let in case of if then else infix infixl infixr foreign forall mdo rec family`,
    lit: `True False Nothing Just Left Right otherwise`,
    type: `IO Int Integer Double Float Char String Bool Maybe Either Functor Foldable mapM_ sequence traverse foldr foldl putStrLn getLine readFile return pure fmap`,
  },

  clojure: {
    aliases: ['clj', 'cljs', 'cljc', 'edn', 'lisp', 'elisp', 'scheme'],
    rules: [
      ['com', R(String.raw`;[^\n]*`)],
      ['str', R(String.raw`"(?:\\[\s\S]|[^"\\])*"?`)],
      ['lit', R(String.raw`:[\w*+!?<>=/-]+`)],
      ['num', R(String.raw`[+-]?\d+\/\d+|[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?M?`)],
      ['id', R(String.raw`[\w*+!\-_$%&|<>=?/.]+`)],
    ],
    kw: `def defn defmacro defmulti defmethod let letfn fn if when when-not cond case do loop recur try catch finally throw ns require require-macros import use quote var set!`,
    lit: `true false nil`,
    type: `map vector list set seq str keyword symbol reduce apply first rest next cons conj assoc dissoc get get-in assoc-in into comp partial identity println print count filter map`,
  },

  swift: {
    aliases: [],
    rules: [
      ['com', R(String.raw`//[^\n]*`)],
      ['com', R(String.raw`/\*[\s\S]*?(?:\*\/|$)`)],
      ['str', R(String.raw`"""[\s\S]*?"""`)],
      ['str', R(String.raw`"(?:\\[\s\S]|[^"\\\n])*"?`)],
      ['var', R(String.raw`[$@][A-Za-z_]\w*`)],
      ['num', R(String.raw`\b0[xX][\da-fA-F_]+\b|\b\d[\d_]*(?:\.\d+)?(?:[eE][+-]?\d+)?\b`)],
      ['id', R(String.raw`[A-Za-z_]\w*`)],
    ],
    kw: `func let var if else guard while for in repeat return switch case default break continue import struct class enum protocol extension init deinit subscript typealias as is nil inout where defer throw throws try catch async await actor some any lazy weak unowned`,
    lit: `true false nil`,
    type: `Int Double Float String Bool Array Dictionary Set Optional Any AnyObject Character Void Data Result print`,
  },

  dart: {
    aliases: [],
    rules: [
      ['com', R(String.raw`//[^\n]*`)],
      ['com', R(String.raw`/\*[\s\S]*?(?:\*\/|$)`)],
      ['str', R(String.raw`r?"(?:\\[\s\S]|[^"\\\n])*"?`)],
      ['var', R(String.raw`[$@][A-Za-z_]\w*`)],
      ['num', R(String.raw`\b0[xX][\da-fA-F]+\b|\b\d[\d_]*(?:\.\d+)?\b`)],
      ['id', R(String.raw`[A-Za-z_$][\w$]*`)],
    ],
    kw: `void var final const class abstract extends implements mixin enum typedef new return if else for while do switch case default break continue try catch finally throw rethrow import export show hide part with as is in async await yield static get set operator required late extension on`,
    lit: `true false null this super`,
    type: `int double num String bool List Map Set Future Stream Widget BuildContext runApp print`,
  },

  zig: {
    aliases: [],
    rules: [
      ['com', R(String.raw`//[^\n]*`)],
      ['str', R(String.raw`"(?:\\[\s\S]|[^"\\\n])*"?`)],
      ['num', R(String.raw`\b0[xXbB][\da-fA-F_]+\b|\b\d[\d_]*(?:\.\d+)?\b`)],
      ['id', R(String.raw`@[A-Za-z_]\w*`)],
      ['id', R(String.raw`[A-Za-z_]\w*`)],
    ],
    kw: `const var fn pub return if else while for switch case break continue defer errdefer try catch orelse unreachable and or struct enum union error comptime inline export extern threadlocal test asm usingnamespace`,
    lit: `true false null undefined`,
    type: `u8 u16 u32 u64 i8 i16 i32 i64 f32 f64 f128 bool void anytype anyerror noreturn type usize isize allocator`,
  },

  julia: {
    aliases: ['jl'],
    rules: [
      ['com', R(String.raw`#[^\n]*`)],
      ['str', R(String.raw`"""[\s\S]*?"""`)],
      ['str', R(String.raw`"(?:\\[\s\S]|[^"\\])*"?`)],
      ['num', R(String.raw`\b0[xX][\da-fA-F]+\b|\b\d[\d_]*(?:\.\d+)?(?:[eEfF][+-]?\d+)?\b`)],
      ['id', R(String.raw`[A-Za-z_!]\w*`)],
    ],
    kw: `function end if elseif else for while break continue return global local const do try catch finally module baremodule using import export struct mutable abstract type where in isa begin quote`,
    lit: `true false nothing missing NaN Inf pi`,
    type: `Int Float64 String Array Vector Matrix Dict Set Tuple Symbol print println length size sum push! map filter reduce zeros ones rand range collect enumerate zip`,
  },

  fsharp: {
    aliases: ['fs', 'fsx'],
    rules: [
      ['com', R(String.raw`\(\*[\s\S]*?\*\)`)],
      ['com', R(String.raw`///[^\n]*`)],
      ['str', R(String.raw`"""[\s\S]*?"""`)],
      ['str', R(String.raw`"[^"\n]*"?`)],
      ['num', R(String.raw`\b0[xX][\da-fA-F]+\b|\b\d[\d.]*\b`)],
      ['id', R(String.raw`[A-Za-z_]\w*'?\b`)],
    ],
    kw: `let rec and in do done if then elif else match with for to downto while try finally raise failwith namespace module open type of val mutable new inherit abstract static override member function return yield use begin end namespace`,
    lit: `true false null None Some Ok Error`,
    type: `int float string bool list array option seq Map Set printfn sprintf failwith raise ignore fst snd List Array Seq String`,
  },

  erlang: {
    aliases: ['erl', 'hrl'],
    rules: [
      ['com', R(String.raw`%[^\n]*`)],
      ['str', R(String.raw`"(?:\\[\s\S]|[^"\\])*"?`)],
      ['lit', R(String.raw`'(?:\\.|[^'\\])'`)],
      ['num', R(String.raw`#b[01]+|#[0-9a-fA-F]+|\b\d[\d_]*\b`)],
      ['id', R(String.raw`\?[A-Za-z_]\w*|[A-Za-z_]\w*`)],
    ],
    kw: `module export import end begin fun case of if receive try catch after when andalso orelse not define spec type record behaviour`,
    lit: `true false undefined ok error`,
    type: `spawn spawn_link self io format lists dicts maps proplists atom_to_list list_to_atom integer_to_list`,
  },

  solidity: {
    aliases: ['sol'],
    rules: [
      ['com', R(String.raw`//[^\n]*`)],
      ['com', R(String.raw`/\*[\s\S]*?(?:\*\/|$)`)],
      ['str', R(String.raw`"(?:\\[\s\S]|[^"\\\n])*"?`)],
      ['str', R(String.raw`'[^'\n]*'?`)],
      ['var', R(String.raw`[$][A-Za-z_]\w*`)],
      ['num', R(String.raw`\b0[xX][\da-fA-F]+\b|\b\d[\d_]*(?:\.\d+)?(?:e\d+)?\b`)],
      ['id', R(String.raw`[A-Za-z_$][\w$]*`)],
    ],
    kw: `pragma solidity contract library interface function constructor modifier event struct enum mapping require revert assert emit public private internal external pure view payable virtual override returns memory storage calldata indexed if else for while do break continue try catch new delete this super using import return is`,
    lit: `true false`,
    type: `address bool string bytes bytes1 bytes4 bytes8 bytes16 bytes32 bytes64 uint uint8 uint16 uint32 uint64 uint128 uint160 uint256 int int8 int32 int64 int128 int256 Block Transaction msg block require`,
  },

  objectivec: {
    aliases: ['objc', 'obj-c', 'mm', 'm'],
    rules: [
      ['com', R(String.raw`//[^\n]*`)],
      ['com', R(String.raw`/\*[\s\S]*?(?:\*\/|$)`)],
      ['str', R(String.raw`@"(?:\\[\s\S]|[^"\\])*"?`)],
      ['str', R(String.raw`"(?:\\[\s\S]|[^"\\\n])*"?`)],
      ['pre', R(String.raw`@[A-Za-z_]\w*`)],
      ['num', R(String.raw`\b0[xX][\da-fA-F]+\b|\b\d[\d_]*(?:\.\d+)?\b`)],
      ['id', R(String.raw`[A-Za-z_]\w*`)],
    ],
    kw: `if else for while switch case default break continue return void int float double char long short signed unsigned const static extern typedef struct union enum @interface @implementation @end @property @synthesize @selector @protocol @class @public @private @protected @autoreleasepool nil YES NO self super id`,
    lit: `YES NO nil`,
    type: `NSString NSArray NSDictionary NSMutableArray NSMutableDictionary NSObject NSInteger CGFloat BOOL instancetype`,
  },

  latex: {
    aliases: ['tex'],
    rules: [
      ['com', R(String.raw`%[^\n]*`)],
      ['kw', R(String.raw`\\[a-zA-Z@]+\*?`)],
      ['str', R(String.raw`"(?:\\[\s\S]|[^"\\])*"?`)],
      ['num', R(String.raw`\b\d+(?:\.\d+)?\b`)],
      ['id', R(String.raw`[A-Za-z]+`)],
    ],
    kw: `documentclass usepackage begin end section subsection subsubsection item label ref cite includegraphics textbf textit maketitle tableofcontents newcommand renewcommand title author date documentstyle`,
    lit: ``,
    type: `textbf texttt emph textsc underline footnote caption`,
  },

  asm: {
    aliases: ['assembly', 'nasm', 'gas', 's'],
    rules: [
      ['com', R(String.raw`[;#][^\n]*`)],
      ['var', R(String.raw`[%@$]\w+`)],
      ['kw', R(String.raw`^[ \t]*[a-z][\w.]*`, 'm')],
      ['str', R(String.raw`"(?:\\[\s\S]|[^"\\])*"?`)],
      ['str', R(String.raw`'[^'\n]*'?`)],
      ['num', R(String.raw`\b0[xX][\da-fA-F]+\b|\b\d+\b`)],
      ['id', R(String.raw`[A-Za-z_.$][\w.$]*`)],
    ],
    kw: `mov movl movq lea push pop add sub mul div imul idiv inc dec jmp call ret cmp test nop hlt int syscall cld std rep repe repne section global extern db dw dd equ times org end bits use16 use32`,
    lit: ``,
    type: `rax rbx rcx rdx rsi rdi rsp rbp eax ebx ecx edx esi edi r8 r9 r10 ax bx cx dx sp bp`,
  },

  protobuf: {
    aliases: ['proto'],
    rules: [
      ['com', R(String.raw`//[^\n]*`)],
      ['com', R(String.raw`/\*[\s\S]*?(?:\*\/|$)`)],
      ['kw', R(String.raw`^[ \t]*(?:message|enum|service|rpc|package|import|option|reserved|extend|repeated|optional|required|oneof|map|returns|stream|public)\b`, 'm')],
      ['str', R(String.raw`"(?:\\[\s\S]|[^"\\])*"?`)],
      ['num', R(String.raw`\b\d+(?:\.\d+)?\b`)],
      ['type', R(String.raw`^[ \t]*[A-Z]\w*`, 'm')],
      ['id', R(String.raw`[a-z_]\w*`)],
    ],
    kw: `message enum service rpc package import option reserved extend repeated optional required oneof map returns stream public syntax to max`,
    lit: `true false`,
    type: `string int32 int64 uint32 uint64 sint32 sint64 fixed32 fixed64 sfixed32 sfixed64 bool bytes double float`,
  },

  crystal: {
    aliases: ['cr'],
    rules: [
      ['com', R(String.raw`#[^\n]*`)],
      ['str', R(String.raw`"(?:\\[\s\S]|[^"\\])*"?`)],
      ['str', R(String.raw`'[^'\n]*'?`)],
      ['var', R(String.raw`@[A-Za-z_]\w*`)],
      ['num', R(String.raw`\b\d[\d_]*(?:\.\d+)?\b`)],
      ['id', R(String.raw`[A-Za-z_][\w?!]*`)],
    ],
    kw: `def end class module struct enum lib fun macro if elsif else unless case when then while until for do break next return yield begin rescue ensure raise require include extend property getter setter abstract`,
    lit: `true false nil`,
    type: `puts print p pp Array Hash String Int Float Nil Bool Tuple NamedTuple`,
  },

  nim: {
    aliases: [],
    rules: [
      ['com', R(String.raw`#[^\n]*`)],
      ['str', R(String.raw`"""[\s\S]*?"""`)],
      ['str', R(String.raw`"(?:\\[\s\S]|[^"\\])*"?`)],
      ['num', R(String.raw`\b0[xXbBoO][\da-fA-F_]+\b|\b\d[\d_]*(?:\.\d+)?\b`)],
      ['id', R(String.raw`[A-Za-z_]\w*`)],
    ],
    kw: `proc func method iterator template type var let const enum object tuple array set seq when if elif else for while case of break continue discard return yield import include export raise try except finally and or not in`,
    lit: `true false nil`,
    type: `int int8 int16 int32 int64 float float32 float64 string char bool seq array list set table variant object echo writeLine`,
  },

  ocaml: {
    aliases: ['ml', 'mli'],
    rules: [
      ['com', R(String.raw`\(\*[\s\S]*?\*\)`)],
      ['str', R(String.raw`"(?:\\[\s\S]|[^"\\])*"?`)],
      ['lit', R(String.raw`'(?:\\.|[^'\\])'`)],
      ['num', R(String.raw`\b0[xX][\da-fA-F]+\b|\b\d[\d.]*\b`)],
      ['id', R(String.raw`[A-Za-z_][\w']*`)],
    ],
    kw: `let rec and in fun function type module struct sig end val external open include exception try raise with match when if then else for while do done new inherit initializer begin`,
    lit: `true false`,
    type: `int float string bool list array option unit List Array String print_endline print_string List.`,
  },

  diff: {
    aliases: ['patch'],
    rules: [
      ['add', R(String.raw`\+[^\n]*`, 'm')],
      ['del', R(String.raw`-[^\n]*`, 'm')],
      ['key', R(String.raw`@@[^\n]*`, 'm')],
    ],
    kw: ``,
    lit: ``,
    type: ``,
  },

  markdown: {
    aliases: ['md', 'mdx'],
    rules: [
      ['key', R(String.raw`#{1,6} [^\n]*`)],
      ['key', R(String.raw`[-*+] [^\n]*`)],
      ['str', R(String.raw`\`[^\`\n]*\`?`)],
      ['str', R(String.raw`\*\*[^*\n]+\*\*`)],
      ['kw', R(String.raw`\[[^\]\n]*\]\([^)\n]*\)`)],
    ],
    kw: ``,
    lit: ``,
    type: ``,
  },
};

/* ---------------- Lookup tables ---------------- */

/** Builds the alias -> canonical map and compiles keyword lookups, once. */
const ALIASES = new Map();
for (const [name, cfg] of Object.entries(LANGS)) {
  ALIASES.set(name, name);
  for (const a of cfg.aliases) ALIASES.set(a, name);
  const lower = (s) => (cfg.ci ? s.toLowerCase() : s);
  cfg.kwSet = new Set(cfg.kw.split(/\s+/).filter(Boolean).map(lower));
  cfg.litSet = new Set(cfg.lit.split(/\s+/).filter(Boolean).map(lower));
  cfg.typeSet = new Set(cfg.type.split(/\s+/).filter(Boolean).map(lower));
}

/** Resolves a fence label to a canonical language name, or null. */
/**
 * Fence labels are captured by /```(\w+)?/, so `#` and `+` never survive to
 * here. This normalisation is for `renderCode` called with a label from another
 * source (and for the common `c#` / `c++` spellings).
 */
const LANG_FIXUPS = new Map([
  ['c#', 'csharp'], ['c++', 'cpp'], ['objective-c', 'c'], ['golang', 'go'],
]);

export function resolveLang(label) {
  if (!label) return null;
  const key = label.trim().toLowerCase().split(/[\s:,]/)[0];
  if (!key) return null;
  return ALIASES.get(key) ?? ALIASES.get(LANG_FIXUPS.get(key) ?? '') ?? null;
}

/* ---------------- Tokenizer ---------------- */

/** Token kind -> class name. */
const SPAN = {
  com: 'c-com', str: 'c-str', num: 'c-num', kw: 'c-kw',
  lit: 'c-lit', fn: 'c-fn', type: 'c-type', var: 'c-var',
  key: 'c-key', tag: 'c-tag', add: 'c-add', del: 'c-del',
  // Compiler directives, command-line flags, and Objective-C message sends
  // all read as "structural punctuation" and share the tag/key colours.
  pre: 'c-tag', flag: 'c-key',
};

/**
 * Highlights `code` and returns `{ html, lang }`, or null when the language is
 * unknown so the caller can fall back to plain text rather than pretending.
 */
export function highlight(code, label) {
  const lang = resolveLang(label);
  if (!lang) return null;
  const cfg = LANGS[lang];
  const lower = (s) => (cfg.ci ? s.toLowerCase() : s);

  const out = [];
  let plain = '';
  let i = 0;

  const flushPlain = () => {
    if (plain) {
      out.push(esc(plain));
      plain = '';
    }
  };
  const emit = (cls, text) => {
    flushPlain();
    out.push(`<span class="${cls}">${esc(text)}</span>`);
  };

  while (i < code.length) {
    let matched = false;

    for (const [kind, regex] of cfg.rules) {
      // Sticky: lastIndex is the only position tried, so this stays linear.
      regex.lastIndex = i;
      const m = regex.exec(code);
      if (!m || !m[0].length) continue;

      const text = m[0];

      if (kind === 'id') {
        // A leading sigil (@annotation, #[] , -flag) colours with its word.
        const bare = text.replace(/^[@#$]+/, '');
        const w = lower(bare);
        if (cfg.kwSet.has(w)) emit(SPAN.kw, text);
        else if (cfg.litSet.has(w)) emit(SPAN.lit, text);
        else if (cfg.typeSet.has(w)) emit(SPAN.type, text);
        // An identifier immediately followed by `(` reads as a call site.
        else if (code[i + text.length] === '(') emit(SPAN.fn, text);
        else {
          flushPlain();
          plain += text;
        }
      } else {
        emit(SPAN[kind] ?? 'c-pun', text);
      }

      i += text.length;
      matched = true;
      break;
    }

    if (!matched) {
      plain += code[i];
      i += 1;
    }
  }

  flushPlain();
  return { html: out.join(''), lang };
}

/** Copy/Download buttons for every rendered block. Static markup (no code or
 *  label text inside the tags themselves), so the escaping selftests are
 *  unaffected — only the visible lang label is escaped per block. */
const codeToolsHtml =
  '<span class="code-tools"><button type="button" class="code-copy">Copy</button>' +
  '<button type="button" class="code-dl">Download</button></span>';

/**
 * The `<pre><code>` markup for a fence, wrapped in a `.codeblock` header
 * carrying the language plus per-block Copy / Download buttons. An unknown or
 * missing language falls back to plain escaped text, so an unlabelled block
 * still renders correctly.
 */
export function renderCode(code, label) {
  const body = code.replace(/\n$/, '');
  const result = highlight(body, label);
  const raw = (label ?? '').trim().split(/[\s:,]/)[0].slice(0, 24);
  const head = (name) =>
    `<div class="code-head"><span class="code-lang">${esc(name)}</span>${codeToolsHtml}</div>`;
  if (!result) {
    return `<div class="codeblock">${head(raw || 'text')}<pre><code data-lang="${escAttr(raw)}">${esc(body)}</code></pre></div>`;
  }
  return `<div class="codeblock">${head(result.lang)}<pre><code class="hl" data-lang="${escAttr(result.lang)}">${result.html}</code></pre></div>`;
}
