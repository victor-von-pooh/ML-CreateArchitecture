import { AlertTriangle, Info, Trash2 } from 'lucide-react'
import { memo, useId, useState } from 'react'

import type { CheckResult, Issue } from '../../api/client'
import { Button } from '../../components/ui'
import { formatCount, formatShape } from '../../lib/format'
import { formatValue, parseShape, parseValue } from '../../lib/params'
import {
  isValidName,
  nodeInputs,
  refOptions,
  removeItem,
  renameItem,
  setPortRefs,
  type ParamSpec,
  type Spec,
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
}: {
  param: ParamSpec
  value: unknown
  onChange: (value: unknown) => void
}) {
  const id = useId()
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
          id={id}
          list={listId}
          defaultValue={formatValue(value)}
          placeholder={param.default !== null ? formatValue(param.default) : ''}
          onChange={(event) => {
            // 候補から選んだ場合 (候補と同じ文字になった場合) はすぐに反映
            const text = event.target.value
            if (param.suggestions?.some((choice) => String(choice) === text)) {
              onChange(parseValue(param, text))
            }
          }}
          onBlur={(event) => onChange(parseValue(param, event.target.value))}
          onKeyDown={(event) =>
            event.key === 'Enter' && onChange(parseValue(param, event.currentTarget.value))
          }
          className="h-7 w-full rounded-md border border-line bg-bg px-2 font-mono text-xs"
        />
        {param.suggestions && listId && <Suggestions id={listId} suggestions={param.suggestions} />}
      </>
    )
  }
  return (
    <div className="grid grid-cols-[minmax(0,8rem)_minmax(0,1fr)] items-center gap-x-2 gap-y-0.5">
      {label}
      {field}
      {hint && <p className="col-start-2 text-[10px] text-ink-3">{hint}</p>}
    </div>
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
  const commit = (value: string) => {
    if (value === name) return setError(null)
    if (!isValidName(value)) return setError('英字か _ で始まり, 英数字と _ だけにしてください')
    if (taken(value)) return setError('同じ名前が既にあります')
    setError(null)
    onRename(value)
  }
  return (
    <div>
      <label htmlFor={id} className="text-[11px] text-ink-3">
        名前
      </label>
      <input
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
            <li key={`${ref}-${index}`} className="flex items-center gap-1 font-mono text-xs">
              <span className="text-ink-3">{index + 1}.</span>
              <span className="min-w-0 flex-1 truncate">{ref}</span>
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
}: {
  spec: Spec
  units: Map<string, UnitSpec>
  check: CheckResult | undefined
  selected: string | null
  onChange: (spec: Spec) => void
  onSelect: (id: string | null) => void
}) {
  const taken = (value: string) => value in spec.inputs || value in spec.nodes
  const issuesFor = (predicate: (issue: Issue) => boolean) =>
    (check?.issues ?? []).filter(predicate)
  const remove = () => {
    if (!selected) return
    onChange(removeItem(spec, units, selected))
    onSelect(null)
  }

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
          key={name}
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
          key={name}
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
          key={selected}
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
          onChange={(value) => {
            const shape = parseShape(value)
            if (shape) update({ shape })
          }}
        />
        <ParamField
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
  const setValue = (field: 'params' | 'forward_params', key: string, value: unknown) => {
    const values = { ...(node[field] ?? {}) }
    if (value === undefined) delete values[key]
    else values[key] = value
    onChange({ ...spec, nodes: { ...spec.nodes, [selected]: { ...node, [field]: values } } })
  }
  return (
    <div className="space-y-4 p-3">
      <NameField
        key={selected}
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
      {result && (
        <div className="rounded-md bg-surface-2 p-2 text-xs">
          {result.status === 'ok' ? (
            <>
              {Object.entries(result.outputs ?? {}).map(([port, shape]) => (
                <div key={port} className="font-mono">
                  {port}: {formatShape(shape)}
                </div>
              ))}
              <div className="text-ink-3">パラメータ数 {formatCount(result.params)}</div>
            </>
          ) : (
            <div className="text-critical-text">{result.error}</div>
          )}
        </div>
      )}
      <IssueList issues={issuesFor((issue) => issue.node === selected)} />
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
      <Button variant="danger" icon={Trash2} onClick={remove}>
        ノードを消す
      </Button>
    </div>
  )
}
