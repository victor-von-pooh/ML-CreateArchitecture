import { AlertTriangle, ArrowDown, ArrowUp, Copy, Info, Plus, Trash2 } from 'lucide-react'
import { memo, useEffect, useId, useRef, useState } from 'react'

import type { CheckResult, Issue } from '../../api/client'
import { Button } from '../../components/ui'
import { failedNodes, formatCount, formatShape } from '../../lib/format'
import { checkValue, formatValue, parseShape, parseValue } from '../../lib/params'
import {
  isValidName,
  nodeInputs,
  refOptions,
  removeItem,
  renameItem,
  setPortRefs,
  type ParamSpec,
  SUBGRAPH_PREFIX,
  incomingValues,
  type Spec,
  type Subgraph,
  type UnitSpec,
} from '../../lib/graph'

// 入力の例に使える型と値の作り方 (ML-ModelControl と同じ)
const DTYPES = ['float32', 'float64', 'float16', 'bfloat16', 'int64', 'int32', 'bool']
const FILLS = ['zeros', 'ones', 'arange']

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
function ParamField({
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
  useEffect(() => {
    const input = inputRef.current
    if (input && document.activeElement !== input) input.value = text
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
function ExtraParams({
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
function NameField({
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
function PortRefs({
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
function IssueList({ issues }: { issues: Issue[] }) {
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
            {issue.location && <span className="font-mono">{issue.location}: </span>}
            {issue.message}
          </span>
        </li>
      ))}
    </ul>
  )
}

/** 選んだノード, 入力, 出力の詳細と編集 */
export function Inspector({
  spec,
  units,
  check,
  selected,
  onChange,
  onSelect,
  onDuplicate,
  onEditSubgraph,
}: {
  spec: Spec
  units: Map<string, UnitSpec>
  check: CheckResult | undefined
  selected: string | null
  onChange: (spec: Spec) => void
  onSelect: (id: string | null) => void
  /** 部品のノードの写しを置く */
  onDuplicate: (name: string) => void
  /** 部分モデルの中身の編集を始める */
  onEditSubgraph?: (name: string) => void
}) {
  const taken = (value: string) => value in spec.inputs || value in spec.nodes
  const issuesFor = (predicate: (issue: Issue) => boolean) =>
    (check?.issues ?? []).filter(predicate)
  const remove = () => {
    if (!selected) return
    onChange(removeItem(spec, units, selected))
    onSelect(null)
  }

  const failed = failedNodes(check, spec)

  // 何も選んでいない場合は, 全体の情報と定義全体の誤り
  if (!selected) {
    return (
      <div className="space-y-3 p-3 text-xs">
        <p className="text-ink-3">
          ノードを選ぶと, 引数と推定した出力の形を確かめて編集できます. 左の一覧から部品を加え,
          口から口へドラッグするか, ノードの「入力のつなぎ先」から選んでつなぎます.
          線とノードは選んで Delete で消せます.
        </p>
        <dl className="grid grid-cols-2 gap-1">
          <dt className="text-ink-3">パラメータ数</dt>
          <dd className="font-mono">{formatCount(check?.params)}</dd>
          <dt className="text-ink-3">学習するもの</dt>
          <dd className="font-mono">{formatCount(check?.trainable)}</dd>
        </dl>
        {check?.estimated_only && (
          <p className="text-warning-text">パラメータ数が上限を超えるので, 形は推定していません.</p>
        )}
        <IssueList issues={issuesFor((issue) => !issue.node)} />
        {failed.length > 0 && (
          <section aria-label="形の推定で失敗したもの" className="space-y-1">
            <h3 className="text-xs font-semibold text-critical-text">
              形の推定で失敗したもの ({failed.length})
            </h3>
            <ul className="space-y-1">
              {failed.map(({ id, label, error }) => (
                <li key={id}>
                  <button
                    type="button"
                    onClick={() => onSelect(id)}
                    className="w-full rounded-md px-1 py-0.5 text-left text-xs hover:bg-surface-2"
                  >
                    <span className="font-mono font-semibold">{label}</span>
                    <span className="block text-critical-text [overflow-wrap:anywhere]">
                      {error}
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          </section>
        )}
      </div>
    )
  }

  // 出力
  if (selected.startsWith('output:')) {
    const name = selected.slice('output:'.length)
    const shape = check?.outputs?.[name]
    return (
      <div className="space-y-3 p-3">
        <NameField
          // 兄弟の欄と重ならない key (重なると別のものを選んだ後も欄が残る)
          key={`name:${name}`}
          name={name}
          taken={(value) => value in spec.outputs || value in (spec.ui?.outputs ?? {})}
          onRename={(value) => {
            const outputs = Object.fromEntries(
              Object.entries(spec.outputs).map(([key, ref]) => [key === name ? value : key, ref]),
            )
            const uiOutputs = Object.fromEntries(
              Object.entries(spec.ui?.outputs ?? {}).map(([key, ui]) => [
                key === name ? value : key,
                ui,
              ]),
            )
            onChange({ ...spec, outputs, ui: { ...spec.ui, outputs: uiOutputs } })
            onSelect(`output:${value}`)
          }}
        />
        <p className="text-xs text-ink-3">
          モデルの出力. 出力が 1 つならその値, 2 つ以上なら名前ごとの辞書を返します.
        </p>
        {name in spec.outputs && (
          <p className="font-mono text-xs">形: {shape ? formatShape(shape) : '?'}</p>
        )}
        <PortRefs
          key={`refs:${name}`}
          port="つなぎ先"
          refs={name in spec.outputs ? [spec.outputs[name]!] : []}
          options={refOptions(spec, units, null)}
          variadic={false}
          optional={false}
          onChange={(refs) => {
            const outputs = { ...spec.outputs }
            const uiOutputs = { ...(spec.ui?.outputs ?? {}) }
            if (refs[0]) outputs[name] = refs[0]
            else {
              delete outputs[name]
              uiOutputs[name] ??= {}
            }
            onChange({ ...spec, outputs, ui: { ...spec.ui, outputs: uiOutputs } })
          }}
        />
        <IssueList issues={issuesFor((issue) => issue.location === `outputs.${name}`)} />
        <Button variant="danger" icon={Trash2} onClick={remove}>
          出力を消す
        </Button>
      </div>
    )
  }

  // 入力
  if (selected in spec.inputs) {
    const input = spec.inputs[selected]!
    const update = (patch: Partial<typeof input>) =>
      onChange({ ...spec, inputs: { ...spec.inputs, [selected]: { ...input, ...patch } } })
    return (
      <div className="space-y-3 p-3">
        <NameField
          key={`name:${selected}`}
          name={selected}
          taken={taken}
          onRename={(value) => {
            onChange(renameItem(spec, selected, value))
            onSelect(value)
          }}
        />
        <p className="text-xs text-ink-3">
          モデルの入力. 形 (バッチの次元も含めて書いたとおり) と型は, 形の推定にだけ使います. 1
          の次元は他の大きさに合ってしまうので避けてください.
        </p>
        <ParamField
          // 入力を選び直したら作り直して値と誤りを合わせる
          key={`${selected}-shape`}
          param={{
            name: 'shape',
            kind: 'list',
            default: null,
            required: true,
            description: '例: [4, 16]',
            minimum: null,
            maximum: null,
            choices: null,
          }}
          value={input.shape}
          check={(text) => (parseShape(text) ? null : '0 以上の整数を並べてください (例: [4, 16])')}
          onChange={(value) => {
            const shape = parseShape(value)
            if (shape) update({ shape })
          }}
        />
        <ParamField
          // 入力を選び直したら作り直して値と誤りを合わせる
          key={`${selected}-dtype`}
          param={{
            name: 'dtype',
            kind: 'str',
            default: 'float32',
            required: false,
            description: '単語の番号や edge_index は int64',
            minimum: null,
            maximum: null,
            choices: DTYPES,
          }}
          value={input.dtype}
          onChange={(value) => update({ dtype: String(value) })}
        />
        <ParamField
          // 入力を選び直したら作り直して値と誤りを合わせる
          key={`${selected}-fill`}
          param={{
            name: 'fill',
            kind: 'str',
            default: 'zeros',
            required: false,
            description: '注意のマスクは ones',
            minimum: null,
            maximum: null,
            choices: FILLS,
          }}
          value={input.fill}
          onChange={(value) => update({ fill: String(value) })}
        />
        <IssueList
          issues={issuesFor(
            (issue) =>
              !issue.node &&
              (issue.location === `inputs.${selected}` ||
                (issue.location ?? '').startsWith(`inputs.${selected}.`)),
          )}
        />
        <Button variant="danger" icon={Trash2} onClick={remove}>
          入力を消す
        </Button>
      </div>
    )
  }

  // 部品のノード
  const node = spec.nodes[selected]
  if (!node) return null
  const unit = units.get(node.unit)
  const result = check?.nodes?.[selected]
  const incoming = incomingValues(spec, check, selected, units)
  const nodeIssues = issuesFor((issue) => issue.node === selected)
  const setValue = (field: 'params' | 'forward_params', key: string, value: unknown) => {
    const values = { ...(node[field] ?? {}) }
    if (value === undefined) delete values[key]
    else values[key] = value
    onChange({ ...spec, nodes: { ...spec.nodes, [selected]: { ...node, [field]: values } } })
  }
  return (
    <div className="space-y-4 p-3">
      <NameField
        key={`name:${selected}`}
        name={selected}
        taken={taken}
        onRename={(value) => {
          onChange(renameItem(spec, selected, value))
          onSelect(value)
        }}
      />
      <div>
        <div className="font-mono text-xs font-semibold">{node.unit}</div>
        {unit?.description && <p className="text-xs text-ink-3">{unit.description}</p>}
      </div>
      {node.unit.startsWith(SUBGRAPH_PREFIX) && (
        <SubgraphContents
          sub={spec.subgraphs?.[node.unit.slice(SUBGRAPH_PREFIX.length)]}
          onEdit={
            onEditSubgraph
              ? () => onEditSubgraph(node.unit.slice(SUBGRAPH_PREFIX.length))
              : undefined
          }
        />
      )}
      {(result || incoming.length > 0) && (
        <div className="space-y-1 rounded-md bg-surface-2 p-2 text-xs">
          {/* 入ってくる形 (形の合わない誤りを, 前のノードを見に行かずに直せるように) */}
          {incoming.length > 0 && (
            <div>
              <div className="text-ink-3">入る形</div>
              {incoming.map(({ port, ref, value }, index) => (
                <div key={`${port}-${ref}-${index}`} className="font-mono break-all">
                  {port} ← {ref}: {formatShape(value)}
                </div>
              ))}
            </div>
          )}
          {result?.status === 'ok' ? (
            <div>
              <div className="text-ink-3">出る形</div>
              {Object.entries(result.outputs ?? {}).map(([port, shape]) => (
                <div key={port} className="font-mono">
                  {port}: {formatShape(shape)}
                </div>
              ))}
              <div className="text-ink-3">パラメータ数 {formatCount(result.params)}</div>
            </div>
          ) : (
            // 確認の誤りは下の一覧に出るので, 確認の誤りが無い場合だけ計算の誤り
            result &&
            !nodeIssues.some((issue) => issue.severity === 'error') && (
              <div className="text-critical-text">{result.error}</div>
            )
          )}
        </div>
      )}
      <IssueList issues={nodeIssues} />
      {(unit?.inputs.length ?? 0) > 0 && (
        <fieldset className="space-y-2">
          <legend className="mb-1 text-xs font-semibold">入力のつなぎ先</legend>
          {unit!.inputs.map((port) => (
            <PortRefs
              key={`${selected}-${port.name}`}
              port={port.name}
              refs={nodeInputs(node, unit)[port.name] ?? []}
              options={refOptions(spec, units, selected)}
              variadic={port.variadic}
              optional={port.optional}
              onChange={(refs) => onChange(setPortRefs(spec, units, selected, port.name, refs))}
            />
          ))}
        </fieldset>
      )}
      {(unit?.params.length ?? 0) > 0 && (
        <fieldset className="space-y-2">
          <legend className="mb-1 text-xs font-semibold">引数</legend>
          {unit!.params.map((param) => (
            <ParamField
              // ノードを選び直したら作り直して値を合わせる
              key={`${selected}-${param.name}`}
              param={param}
              value={node.params?.[param.name]}
              onChange={(value) => setValue('params', param.name, value)}
            />
          ))}
        </fieldset>
      )}
      {unit?.extra_params && (
        <ExtraParams
          // 兄弟の名前の欄 (key は selected) と重ならない key
          key={`extra:${selected}`}
          known={new Set(unit.params.map((param) => param.name))}
          values={node.params ?? {}}
          onChange={(key, value) => setValue('params', key, value)}
        />
      )}
      {(unit?.forward_params.length ?? 0) > 0 && (
        <fieldset className="space-y-2">
          <legend className="mb-1 text-xs font-semibold">呼び出すときの設定</legend>
          {unit!.forward_params.map((param) => (
            <ParamField
              key={`${selected}-${param.name}`}
              param={param}
              value={node.forward_params?.[param.name]}
              onChange={(value) => setValue('forward_params', param.name, value)}
            />
          ))}
        </fieldset>
      )}
      <Button
        variant="secondary"
        icon={Copy}
        onClick={() => onDuplicate(selected)}
        title="引数ごと写したノードを右下に置く (Ctrl+D. Ctrl+C と Ctrl+V でも写せる)"
        aria-keyshortcuts="Control+D Meta+D"
      >
        複製
      </Button>
      <Button variant="danger" icon={Trash2} onClick={remove}>
        ノードを消す
      </Button>
    </div>
  )
}

/** 部分モデルの中身 (入力の想定の形, 中のノードの部品と引数, 出力) を読むだけの一覧 */
function SubgraphContents({
  sub,
  onEdit,
}: {
  sub: Subgraph | undefined
  /** 部分モデルの中身をキャンバスで編集する */
  onEdit?: () => void
}) {
  if (!sub) return null
  const params = (values: Record<string, unknown> | undefined) =>
    Object.entries(values ?? {})
      .map(([key, value]) => `${key}=${JSON.stringify(value)}`)
      .join(', ')
  return (
    <div className="space-y-2">
      {onEdit && (
        <Button variant="secondary" onClick={onEdit}>
          部分モデルの中身を編集
        </Button>
      )}
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
