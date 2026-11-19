import clsx from 'clsx'
import { Copy, Trash2 } from 'lucide-react'

import type { CheckResult, Issue, WeightsResult } from '../../api/client'
import { Button } from '../../components/ui'
import { failedNodes, formatCount, formatShape } from '../../lib/format'
import { parseShape } from '../../lib/params'
import {
  nodeInputs,
  refOptions,
  removeItem,
  renameItem,
  setPortRefs,
  SUBGRAPH_PREFIX,
  type Spec,
  type UnitSpec,
} from '../../lib/graph'
import { incomingValues } from '../../lib/subgraph'
import { NodeState } from './NodeState'
import { PretrainedLicense } from './PretrainedLicense'
import { CustomModuleHint } from './CustomModuleHint'

// 入力の例に使える型と値の作り方 (ML-ModelControl と同じ)

import { ExtraParams, IssueList, NameField, ParamField, PortRefs, SubgraphContents } from './fields'
import { friendlyMessage } from '../../lib/issues'
const DTYPES = ['float32', 'float64', 'float16', 'bfloat16', 'int64', 'int32', 'bool']
const FILLS = ['zeros', 'ones', 'arange']

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
  onForkSubgraph,
  forkable,
  onReadWeights,
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
  /** ノードが使う部分モデルを写して, そのノードだけの部分モデルにする */
  onForkSubgraph?: (node: string) => void
  /** 2 つ以上のノードが使う部分モデルの名前 (写す操作を出すもの) */
  forkable?: Set<string>
  /** 定義全体を組み立てて外部の重みを読む (報告を返す) */
  onReadWeights?: () => Promise<WeightsResult>
}) {
  const taken = (value: string) => value in spec.inputs || value in spec.nodes
  const issuesFor = (predicate: (issue: Issue) => boolean) =>
    (check?.issues ?? []).filter(predicate)
  const remove = () => {
    if (!selected) return
    onChange(removeItem(spec, units, selected))
    onSelect(null)
  }

  // 確認の誤りか注意のあるノードと, その文 (選ばないと見えないため)

  const nodeWarnings = Object.entries(
    (check?.issues ?? []).reduce<Record<string, { error: boolean; messages: string[] }>>(
      (found, issue) => {
        if (issue.node && issue.node in spec.nodes) {
          const entry = found[issue.node] ?? { error: false, messages: [] }
          entry.messages.push(friendlyMessage(issue))
          entry.error ||= issue.severity === 'error'
          found[issue.node] = entry
        }
        return found
      },
      {},
    ),
  )
    // 誤りのあるノードを先に
    .sort(([, a], [, b]) => Number(b.error) - Number(a.error))
    .map(([node, { error, messages }]) => ({ node, error, messages }))

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
        <p className="text-ink-3">
          口 (ノードの左右の点) は, 左が入る口, 右が出る口. 名前の後の{' '}
          <span className="font-mono">?</span> は省略できる口, <span className="font-mono">…</span>{' '}
          は複数つなげる口. 引数の <span className="text-critical-text">*</span> は必須.
        </p>
        <dl className="grid grid-cols-2 gap-1">
          <dt className="text-ink-3">パラメータ数</dt>
          <dd className="font-mono">{formatCount(check?.params)}</dd>
          <dt className="text-ink-3">学習するパラメータ数</dt>
          <dd className="font-mono">{formatCount(check?.trainable)}</dd>
        </dl>
        {check?.estimated_only && (
          <p className="text-warning-text">パラメータ数が上限を超えるので, 形は推定していません.</p>
        )}
        <IssueList issues={issuesFor((issue) => !issue.node)} />
        <CustomModuleHint issues={issuesFor((issue) => !issue.node)} />
        {/* ノードの誤りと注意 (引数の不足, 出力に届かない, つなぎ忘れなど) は,
            選ばないと見えないので一覧にする (押すと選ぶ) */}
        {nodeWarnings.length > 0 && (
          <section aria-label="誤りと注意のあるノード" className="space-y-1">
            <h3 className="text-xs font-semibold text-ink-2">
              誤りと注意のあるノード ({nodeWarnings.length})
            </h3>
            <ul className="space-y-1">
              {nodeWarnings.map(({ node, error, messages }) => (
                <li key={node}>
                  <button
                    type="button"
                    onClick={() => onSelect(node)}
                    className="w-full rounded-md px-1 py-0.5 text-left text-xs hover:bg-surface-2"
                  >
                    <span className="font-mono font-semibold">{node}</span>
                    {messages.map((message, index) => (
                      <span
                        key={index}
                        className={clsx(
                          'block [overflow-wrap:anywhere]',
                          error ? 'text-critical-text' : 'text-ink-2',
                        )}
                      >
                        {message}
                      </span>
                    ))}
                  </button>
                </li>
              ))}
            </ul>
          </section>
        )}
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
          check={(text) =>
            parseShape(text)
              ? null
              : '0 以上の整数を並べてください (例: [4, 16] か (4, 16). 括弧は閉じる)'
          }
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
            description: 'アテンションのマスクは ones',
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
      {/* 学習済みの重みを読める部品は, 配布元から読んだライセンスと警告 */}
      {(unit?.weights_params?.length ?? 0) > 0 && (
        <PretrainedLicense
          key={`license:${selected}`}
          unit={node.unit}
          params={node.params ?? {}}
          weightsParams={unit?.weights_params ?? []}
        />
      )}
      {node.unit.startsWith(SUBGRAPH_PREFIX) && (
        <SubgraphContents
          sub={spec.subgraphs?.[node.unit.slice(SUBGRAPH_PREFIX.length)]}
          onFork={
            onForkSubgraph && forkable?.has(node.unit.slice(SUBGRAPH_PREFIX.length))
              ? () => onForkSubgraph(selected)
              : undefined
          }
          onEdit={
            onEditSubgraph
              ? () => onEditSubgraph(node.unit.slice(SUBGRAPH_PREFIX.length))
              : undefined
          }
        />
      )}
      {/* 中身が無い場合 (確認の誤りで計算しないノードなど) は欄を出さない */}
      {(incoming.length > 0 ||
        result?.status === 'ok' ||
        (result && !nodeIssues.some((issue) => issue.severity === 'error'))) && (
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
      <CustomModuleHint issues={nodeIssues} />
      {(unit?.inputs.length ?? 0) > 0 && (
        <fieldset className="space-y-2">
          <legend className="mb-1 text-xs font-semibold">入力のつなぎ先</legend>
          {unit!.inputs.map((port) => (
            <PortRefs
              key={`${selected}-${port.name}`}
              port={port.name}
              description={port.description}
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
        <fieldset className="space-y-2" data-first-field>
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
      <NodeState
        key={`state:${selected}`}
        spec={spec}
        name={selected}
        node={node}
        onChange={onChange}
        onReadWeights={onReadWeights}
      />
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
