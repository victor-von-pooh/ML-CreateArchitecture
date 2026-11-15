import { AlertTriangle, ArrowDown, ArrowUp, Info, Plus, Trash2 } from 'lucide-react'
import { memo, useEffect, useId, useRef, useState } from 'react'

import type { Issue } from '../../api/client'
import { friendlyLocation, friendlyMessage } from '../../lib/issues'
import { Button } from '../../components/ui'
import { checkValue, formatValue, parseValue } from '../../lib/params'
import { isValidName, type ParamSpec, type Subgraph } from '../../lib/graph'

// 入力の例に使える型と値の作り方 (ML-ModelControl と同じ)

// 詳細の欄の部品 (引数, 名前, つなぎ先, 誤りの一覧, 部分モデルの中身)

/** 入力欄の値の候補 (数が多いので, 候補が変わらない限り描き直さない) */
const Suggestions = memo(function Suggestions({
  id,
  suggestions,
}: {
  id: string
  suggestions: unknown[]
}) {
  return (
    <datalist id={id}>
      {suggestions.map((choice) => (
        <option key={String(choice)} value={String(choice)} />
      ))}
    </datalist>
  )
})

/** 1 つの引数の入力欄 (真偽値はチェック, 選択肢は選択, それ以外は文字) */
export function ParamField({
  param,
  value,
  onChange,
  check = (text) => checkValue(param, text),
  format = formatValue,
}: {
  param: ParamSpec
  value: unknown
  onChange: (value: unknown) => void
  /** 入力欄の文字の確認 (合わない場合は理由. その値は定義に反映しない) */
  check?: (text: string) => string | null
  /** 値を欄の文字にする (一覧に無い引数では null も文字で見せる) */
  format?: (value: unknown) => string
}) {
  const id = useId()
  const errorId = `${id}-error`
  // 反映しなかった値の誤り (元に戻すなどで値が変わったら消す)
  const [failure, setFailure] = useState<{ message: string } | null>(null)
  // 確かめてから反映 (合わない値は理由を出して反映しない)
  const commit = (text: string) => {
    const message = check(text)
    // 反映しない場合は, 定義に残る値も示す (欄の文字と定義の値が違うため)
    setFailure(
      message === null
        ? null
        : {
            message:
              value === undefined ? message : `${message} (定義は前の値 ${format(value)} のまま)`,
          },
    )
    // 値が変わらない場合は反映しない (元に戻すの履歴に残さないため)
    if (message === null) {
      const parsed = parseValue(param, text)
      if (JSON.stringify(parsed) !== JSON.stringify(value)) onChange(parsed)
    }
  }
  const hintId = `${id}-hint`
  // 欄の外で値が変わった場合 (元に戻すなど) は, 打っている途中でなければ
  // 欄の文字を定義の値に合わせる
  const inputRef = useRef<HTMLInputElement>(null)
  const text = format(value)
  // 欄の外で値が変わったら, 前の誤りを消す
  const [seen, setSeen] = useState(text)
  if (seen !== text) {
    setSeen(text)
    setFailure(null)
  }
  const error = failure?.message ?? null
  // (欄にいても, 打ちかけの文字が無ければ合わせる. 欄の中の Ctrl+Z で定義を
  // 戻した場合も欄に出すため)
  const committed = useRef(text)
  useEffect(() => {
    const input = inputRef.current
    if (input && (document.activeElement !== input || input.value === committed.current)) {
      input.value = text
    }
    committed.current = text
  }, [text])
  const hint = [
    param.description,
    param.minimum !== null ? `${param.minimum} 以上` : '',
    param.maximum !== null ? `${param.maximum} 以下` : '',
    value === undefined && param.default !== null && !param.required
      ? `既定 ${formatValue(param.default)}`
      : '',
  ]
    .filter(Boolean)
    .join(' · ')
  const label = (
    <label htmlFor={id} className="font-mono text-xs text-ink-2">
      {param.name}
      {param.required && <span className="text-critical-text"> *</span>}
    </label>
  )
  let field
  if (param.kind === 'bool') {
    field = (
      <input
        id={id}
        type="checkbox"
        checked={Boolean(value ?? param.default)}
        onChange={(event) => onChange(event.target.checked)}
        className="accent-[var(--accent)]"
      />
    )
  } else if (isOptionalBool(param)) {
    // 真偽値か null (未設定の意味の既定) の引数は, 3 つから選ぶ
    field = (
      <select
        id={id}
        value={value === undefined ? '' : String(value)}
        onChange={(event) =>
          onChange(
            event.target.value === ''
              ? undefined
              : event.target.value === 'null'
                ? null
                : event.target.value === 'true',
          )
        }
        className="h-7 w-full rounded-md border border-line bg-bg px-1 font-mono text-xs"
      >
        <option value="">
          (未設定{param.default !== null ? `. 既定 ${formatValue(param.default)}` : ''})
        </option>
        <option value="true">true</option>
        <option value="false">false</option>
        <option value="null">null</option>
      </select>
    )
  } else if (param.choices) {
    field = (
      <select
        id={id}
        value={value === undefined ? '' : String(value)}
        onChange={(event) =>
          onChange(
            event.target.value === ''
              ? undefined
              : param.choices!.find((choice) => String(choice) === event.target.value),
          )
        }
        className="h-7 w-full rounded-md border border-line bg-bg px-1 font-mono text-xs"
      >
        <option value="">(未設定)</option>
        {param.choices.map((choice) => (
          <option key={String(choice)} value={String(choice)}>
            {String(choice)}
          </option>
        ))}
      </select>
    )
  } else {
    // 候補がある引数 (モデルの名前など) は, 打ちながら候補から選べる
    const listId = param.suggestions ? `${id}-suggestions` : undefined
    field = (
      <>
        <input
          ref={inputRef}
          id={id}
          list={listId}
          defaultValue={text}
          // 打ちかけの文字が無い間の Ctrl+Z は定義を戻す (画面の操作と同じ)
          data-committed={text}
          placeholder={param.default !== null ? formatValue(param.default) : ''}
          onChange={(event) => {
            // 候補の一覧から選んだ場合だけすぐに反映 (打っている途中で候補と
            // 同じ文字になった場合は, Enter か欄を離れたときに反映)
            const text = event.target.value
            const picked =
              !(event.nativeEvent instanceof InputEvent) ||
              event.nativeEvent.inputType === 'insertReplacementText'
            if (picked && param.suggestions?.some((choice) => String(choice) === text)) commit(text)
          }}
          onBlur={(event) => commit(event.target.value)}
          onKeyDown={(event) => event.key === 'Enter' && commit(event.currentTarget.value)}
          aria-invalid={error !== null}
          aria-describedby={
            [error ? errorId : '', hint ? hintId : ''].join(' ').trim() || undefined
          }
          className="h-7 w-full rounded-md border border-line bg-bg px-2 font-mono text-xs aria-[invalid=true]:border-critical"
        />
        {param.suggestions && listId && <Suggestions id={listId} suggestions={param.suggestions} />}
      </>
    )
  }
  return (
    <div className="grid grid-cols-[minmax(0,8rem)_minmax(0,1fr)] items-center gap-x-2 gap-y-0.5">
      {label}
      {field}
      {error && (
        <p id={errorId} role="alert" className="col-start-2 text-[11px] text-critical-text">
          {error}
        </p>
      )}
      {hint && (
        <p id={hintId} className="col-start-2 text-[10px] text-ink-3">
          {hint}
        </p>
      )}
    </div>
  )
}

// Python の予約語 (一覧に無い引数の名前に使えない. 書き出すコードの引数になるため)
const PYTHON_KEYWORDS = new Set([
  'False',
  'None',
  'True',
  'and',
  'as',
  'assert',
  'async',
  'await',
  'break',
  'class',
  'continue',
  'def',
  'del',
  'elif',
  'else',
  'except',
  'finally',
  'for',
  'from',
  'global',
  'if',
  'import',
  'in',
  'is',
  'lambda',
  'nonlocal',
  'not',
  'or',
  'pass',
  'raise',
  'return',
  'try',
  'while',
  'with',
  'yield',
])

/**
 * 一覧に無い引数 (**kwargs で受け取る部品の, einops の式の大きさ h=2 など)
 * の欄. 名前を書いて加え, 値は JSON (読めなければ文字) として読む
 */
export function ExtraParams({
  known,
  values,
  onChange,
}: {
  known: Set<string>
  values: Record<string, unknown>
  onChange: (key: string, value: unknown) => void
}) {
  const id = useId()
  const [draft, setDraft] = useState('')
  const [error, setError] = useState<string | null>(null)
  // 外した後は加える欄に移る (外した行にあった操作の場所を失わないため)
  const draftRef = useRef<HTMLInputElement>(null)
  const extra = Object.entries(values).filter(([key]) => !known.has(key))
  const removeParam = (key: string) => {
    onChange(key, undefined)
    draftRef.current?.focus()
  }
  const add = () => {
    const key = draft.trim()
    if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(key))
      return setError('英字か _ で始まる名前にしてください')
    if (PYTHON_KEYWORDS.has(key)) return setError('Python の予約語は使えません')
    if (known.has(key) || key in values) return setError('その引数は既にあります')
    setError(null)
    setDraft('')
    onChange(key, 0)
  }
  return (
    <fieldset className="space-y-2">
      <legend className="mb-1 text-xs font-semibold">一覧に無い引数</legend>
      {extra.map(([key, value]) => (
        <div key={key} className="flex items-start gap-1">
          <div className="min-w-0 flex-1">
            <ParamField
              param={{
                name: key,
                kind: 'any',
                default: null,
                required: false,
                description: '',
                minimum: null,
                maximum: null,
                choices: null,
              }}
              value={value}
              onChange={(next) => onChange(key, next)}
              // 空は外すのではなく誤り (外すのはごみ箱), null は文字で見せて残す
              check={(text) =>
                text.trim() === '' ? '値を入れてください (外す場合はごみ箱)' : null
              }
              format={(item) => (item === null ? 'null' : formatValue(item))}
            />
          </div>
          <Button
            variant="ghost"
            icon={Trash2}
            onClick={() => removeParam(key)}
            aria-label={`引数 ${key} を外す`}
            title={`引数 ${key} を外す`}
          />
        </div>
      ))}
      <div className="flex items-center gap-1">
        <label htmlFor={id} className="sr-only">
          加える引数の名前
        </label>
        <input
          ref={draftRef}
          id={id}
          value={draft}
          placeholder="引数の名前 (例: h)"
          onChange={(event) => {
            setDraft(event.target.value)
            setError(null)
          }}
          // 日本語の変換を確定する Enter では加えない
          onKeyDown={(event) => event.key === 'Enter' && !event.nativeEvent.isComposing && add()}
          aria-invalid={error !== null}
          aria-describedby={error ? `${id}-error` : undefined}
          className="h-7 min-w-0 flex-1 rounded-md border border-line bg-bg px-2 font-mono text-xs aria-[invalid=true]:border-critical"
        />
        <Button variant="secondary" icon={Plus} onClick={add}>
          加える
        </Button>
      </div>
      {error && (
        <p id={`${id}-error`} role="alert" className="text-[11px] text-critical-text">
          {error}
        </p>
      )}
    </fieldset>
  )
}

/** 名前の入力欄 (規則と重なりを確かめてから変える) */
export function NameField({
  name,
  taken,
  onRename,
}: {
  name: string
  taken: (value: string) => boolean
  onRename: (value: string) => void
}) {
  const id = useId()
  const [error, setError] = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)
  const commit = (value: string) => {
    if (value === name) return setError(null)
    if (!isValidName(value)) return setError('英字か _ で始まり, 英数字と _ だけにしてください')
    if (taken(value)) return setError('同じ名前が既にあります')
    setError(null)
    onRename(value)
    // 名前を変えると欄を作り直すので, 新しい名前の欄にフォーカスを戻す
    if (document.activeElement === inputRef.current) {
      window.requestAnimationFrame(() =>
        document.querySelector<HTMLInputElement>('[data-name-field]')?.focus(),
      )
    }
  }
  return (
    <div>
      <label htmlFor={id} className="text-[11px] text-ink-3">
        名前
      </label>
      <input
        ref={inputRef}
        data-name-field
        id={id}
        key={name}
        defaultValue={name}
        onBlur={(event) => commit(event.target.value)}
        onKeyDown={(event) => event.key === 'Enter' && commit(event.currentTarget.value)}
        aria-invalid={error !== null}
        className="h-8 w-full rounded-md border border-line bg-bg px-2 font-mono text-sm font-semibold aria-[invalid=true]:border-critical"
      />
      {error && <p className="text-[11px] text-critical-text">{error}</p>}
    </div>
  )
}

/** 口のつなぎ先を選ぶ欄 (線を引かずに, キーボードでもつなげる) */
/** リストの 2 つの要素を入れ替えたリスト */
function swap<T>(items: T[], a: number, b: number): T[] {
  const next = [...items]
  ;[next[a], next[b]] = [next[b] as T, next[a] as T]
  return next
}

/** 入力の口のつなぎ先の欄 (1 つは選択, 複数は並びと追加の選択) */
export function PortRefs({
  port,
  refs,
  options,
  variadic,
  optional,
  onChange,
}: {
  port: string
  refs: string[]
  options: { value: string; label: string }[]
  variadic: boolean
  optional: boolean
  onChange: (refs: string[]) => void
}) {
  const id = useId()
  // つなぎ先を 1 つ上か下へ動かし, 動かした先の行の同じ向き (端なら逆向き) の
  // ボタンにフォーカスを移す
  const move = (index: number, step: -1 | 1) => {
    const target = index + step
    if (refs[index] === refs[target]) return
    onChange(swap(refs, index, target))
    const edge = target === 0 || target === refs.length - 1
    const direction = (step === -1) !== edge ? 'up' : 'down'
    window.requestAnimationFrame(() =>
      document.querySelector<HTMLElement>(`[data-move="${id}-${target}-${direction}"]`)?.focus(),
    )
  }
  const label = (
    <label htmlFor={id} className="font-mono text-xs text-ink-2">
      {port}
      {!optional && <span className="text-critical-text"> *</span>}
    </label>
  )
  // 1 つだけつなぐ口は選択, 複数つなぐ口は並びと追加の選択
  if (!variadic) {
    return (
      <div className="grid grid-cols-[minmax(0,8rem)_minmax(0,1fr)] items-center gap-2">
        {label}
        <select
          id={id}
          value={refs[0] ?? ''}
          onChange={(event) => onChange(event.target.value ? [event.target.value] : [])}
          className="h-7 w-full rounded-md border border-line bg-bg px-1 font-mono text-xs"
        >
          <option value="">(未接続)</option>
          {refs[0] && !options.some((option) => option.value === refs[0]) && (
            <option value={refs[0]}>{refs[0]} (候補に無い書き方)</option>
          )}
          {options.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      </div>
    )
  }
  return (
    <div className="grid grid-cols-[minmax(0,8rem)_minmax(0,1fr)] items-start gap-2">
      {label}
      <div className="space-y-1">
        <ol className="space-y-1">
          {refs.map((ref, index) => (
            // 並べ替えても行を作り直さない (押したボタンのフォーカスを保つ)
            <li key={index} className="flex items-center gap-1 font-mono text-xs">
              <span className="text-ink-3">{index + 1}.</span>
              <span className="min-w-0 flex-1 truncate">{ref}</span>
              {/* 順に意味がある口 (Concat など) のための並べ替え */}
              <button
                type="button"
                disabled={index === 0}
                data-move={`${id}-${index}-up`}
                onClick={() => move(index, -1)}
                aria-label={`${port} の ${index + 1} 番目のつなぎ先 ${ref} を上へ`}
                title="上へ"
                className="rounded px-1 text-ink-3 hover:bg-surface-2 hover:text-ink disabled:opacity-30"
              >
                <ArrowUp className="size-3" aria-hidden />
              </button>
              <button
                type="button"
                disabled={index === refs.length - 1}
                data-move={`${id}-${index}-down`}
                onClick={() => move(index, 1)}
                aria-label={`${port} の ${index + 1} 番目のつなぎ先 ${ref} を下へ`}
                title="下へ"
                className="rounded px-1 text-ink-3 hover:bg-surface-2 hover:text-ink disabled:opacity-30"
              >
                <ArrowDown className="size-3" aria-hidden />
              </button>
              <button
                type="button"
                onClick={() => onChange(refs.filter((_, position) => position !== index))}
                aria-label={`${port} の ${index + 1} 番目のつなぎ先 ${ref} を外す`}
                title="外す"
                className="rounded px-1 text-ink-3 hover:bg-surface-2 hover:text-critical-text"
              >
                <Trash2 className="size-3" aria-hidden />
              </button>
            </li>
          ))}
        </ol>
        <select
          id={id}
          value=""
          aria-label={`${port} につなぎ先を加える`}
          onChange={(event) => event.target.value && onChange([...refs, event.target.value])}
          className="h-7 w-full rounded-md border border-line bg-bg px-1 font-mono text-xs"
        >
          <option value="">(つなぎ先を加える)</option>
          {options.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      </div>
    </div>
  )
}

/** 誤りと注意の一覧 */
export function IssueList({ issues }: { issues: Issue[] }) {
  if (issues.length === 0) return null
  return (
    <ul className="space-y-1">
      {issues.map((issue, index) => (
        <li
          key={`${issue.code}-${index}`}
          className={
            issue.severity === 'error'
              ? 'flex gap-1.5 text-xs text-critical-text'
              : 'flex gap-1.5 text-xs text-warning-text'
          }
        >
          {issue.severity === 'error' ? (
            <AlertTriangle className="mt-0.5 size-3.5 shrink-0" aria-hidden />
          ) : (
            <Info className="mt-0.5 size-3.5 shrink-0" aria-hidden />
          )}
          <span>
            <span className="sr-only">{issue.severity === 'error' ? '誤り' : '注意'}: </span>
            {friendlyLocation(issue) && (
              <span className="font-mono">{friendlyLocation(issue)}: </span>
            )}
            {friendlyMessage(issue)}
          </span>
        </li>
      ))}
    </ul>
  )
}

/** 部分モデルの中身 (入力の想定の形, 中のノードの部品と引数, 出力) を読むだけの一覧 */
export function SubgraphContents({
  sub,
  onEdit,
  onFork,
}: {
  sub: Subgraph | undefined
  /** 部分モデルの中身をキャンバスで編集する */
  onEdit?: () => void
  /** このノードだけが使う部分モデルにする (写して別の部分モデルに) */
  onFork?: () => void
}) {
  if (!sub) return null
  const params = (values: Record<string, unknown> | undefined) =>
    Object.entries(values ?? {})
      .map(([key, value]) => `${key}=${JSON.stringify(value)}`)
      .join(', ')
  return (
    <div className="space-y-2">
      <div className="flex flex-wrap gap-2">
        {onEdit && (
          <Button variant="secondary" onClick={onEdit}>
            部分モデルの中身を編集
          </Button>
        )}
        {onFork && (
          <Button
            variant="ghost"
            onClick={onFork}
            title="同じ部分モデルを使う他のノードはそのままで, このノードだけ中身を変えられるように写す"
          >
            このノードだけの部分モデルにする
          </Button>
        )}
      </div>
      <details className="rounded-md border border-line p-2 text-xs">
        <summary className="cursor-pointer font-semibold text-ink-2">部分モデルの中身</summary>
        <dl className="mt-2 space-y-1">
          <dt className="text-ink-3">入力 (元の定義の形)</dt>
          {Object.entries(sub.inputs ?? {}).map(([name, input]) => (
            <dd key={name} className="font-mono">
              {name}
              {input?.shape ? `: [${input.shape.join(', ')}]` : ''}
            </dd>
          ))}
          <dt className="text-ink-3">ノード</dt>
          {Object.entries(sub.nodes ?? {}).map(([name, child]) => (
            <dd key={name} className="font-mono break-all">
              {name}: {child.unit}
              {child.params && Object.keys(child.params).length > 0 && (
                <span className="text-ink-3"> ({params(child.params)})</span>
              )}
            </dd>
          ))}
          <dt className="text-ink-3">出力</dt>
          {Object.entries(sub.outputs ?? {}).map(([name, ref]) => (
            <dd key={name} className="font-mono">
              {name} ← {ref}
            </dd>
          ))}
        </dl>
      </details>
    </div>
  )
}

/** 真偽値か null だけを受け取る引数か (Optional[bool] の型ヒント) */
function isOptionalBool(param: ParamSpec): boolean {
  const types = param.types
  return (
    param.kind === 'any' &&
    Array.isArray(types) &&
    types.includes('bool') &&
    types.every((type) => type === 'bool' || type === 'null')
  )
}
