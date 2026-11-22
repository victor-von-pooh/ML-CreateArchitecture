import { useMutation } from '@tanstack/react-query'
import clsx from 'clsx'
import { Download, ExternalLink, Plus, Trash2 } from 'lucide-react'
import { useEffect, useId, useRef, useState } from 'react'

import type { WeightsReport, WeightsResult } from '../../api/client'
import { Button } from '../../components/ui'
import type { NodeDef, RemapRule, Spec, WeightsDef } from '../../lib/graph'
import {
  formatShareLines,
  parseNames,
  parseShareLines,
  ruleKind,
  setNodeState,
  setNodeStates,
  shareSources,
  switchRuleKind,
  switchWeightsSource,
  weightsSource,
  type WeightsSource,
} from '../../lib/state'
import { LicenseCard } from './PretrainedLicense'

const LABEL = 'text-[11px] text-ink-3'
const INPUT =
  'h-7 w-full rounded-md border border-line bg-bg px-2 font-mono text-xs aria-[invalid=true]:border-critical'
const SELECT = 'h-7 w-full rounded-md border border-line bg-bg px-1 text-xs'

/** 打ち終えたら (欄を離れるか Enter) 反映する 1 行の文字の欄 */
function TextField({
  label,
  value,
  placeholder,
  check,
  onCommit,
}: {
  label: string
  value: string
  placeholder?: string
  /** 合わない文字の理由 (合わなければ反映しない) */
  check?: (text: string) => string | null
  onCommit: (text: string) => void
}) {
  const id = useId()
  const errorId = `${id}-error`
  const [error, setError] = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)
  // 欄の外で値が変わったら (元に戻すなど), 前の誤りを消し, 打ちかけの文字が
  // 無ければ欄に出す (作り直さないので Enter の後も focus が残る)
  const [seen, setSeen] = useState(value)
  if (seen !== value) {
    setSeen(value)
    setError(null)
  }
  const committed = useRef(value)
  useEffect(() => {
    const input = inputRef.current
    if (input && (document.activeElement !== input || input.value === committed.current)) {
      input.value = value
    }
    committed.current = value
  }, [value])
  const commit = (text: string) => {
    const message = check?.(text) ?? null
    setError(message)
    if (message === null && text !== value) onCommit(text)
  }
  return (
    <div>
      <label htmlFor={id} className={LABEL}>
        {label}
      </label>
      <input
        ref={inputRef}
        id={id}
        defaultValue={value}
        data-committed={value}
        placeholder={placeholder}
        onBlur={(event) => commit(event.target.value)}
        onKeyDown={(event) => event.key === 'Enter' && commit(event.currentTarget.value)}
        aria-invalid={error !== null}
        aria-describedby={error ? errorId : undefined}
        className={INPUT}
      />
      {error && (
        <p id={errorId} className="text-[11px] text-critical-text">
          {error}
        </p>
      )}
    </div>
  )
}

/** チェックの欄 */
function CheckField({
  label,
  checked,
  hint,
  onChange,
}: {
  label: string
  checked: boolean
  hint?: string
  onChange: (checked: boolean) => void
}) {
  return (
    <label className="flex items-start gap-1.5 text-xs">
      <input
        type="checkbox"
        checked={checked}
        onChange={(event) => onChange(event.target.checked)}
        className="mt-0.5"
      />
      <span>
        {label}
        {hint && <span className="block text-[11px] text-ink-3">{hint}</span>}
      </span>
    </label>
  )
}

/** 選ぶ欄 */
function SelectField<T extends string>({
  label,
  value,
  options,
  onChange,
}: {
  label: string
  value: T
  /** 値, 見せる文字, 選べないか */
  options: [T, string, boolean?][]
  onChange: (value: T) => void
}) {
  const id = useId()
  return (
    <div>
      <label htmlFor={id} className={LABEL}>
        {label}
      </label>
      <select
        id={id}
        value={value}
        onChange={(event) => onChange(event.target.value as T)}
        className={SELECT}
      >
        {options.map(([key, text, disabled]) => (
          <option key={key} value={key} disabled={disabled}>
            {text}
          </option>
        ))}
      </select>
    </div>
  )
}

/** ノードのパラメータの状態 (定義の版 2. 凍結, 評価の形, 零初期化, 共有, 外部の重み) の欄 */
export function NodeState({
  spec,
  name,
  node,
  onChange,
  onReadWeights,
}: {
  spec: Spec
  name: string
  node: NodeDef
  onChange: (spec: Spec) => void
  /** 定義全体を組み立てて外部の重みを読む (無ければボタンを出さない) */
  onReadWeights?: () => Promise<WeightsResult>
}) {
  const set = <K extends 'train' | 'eval_mode' | 'zero_init' | 'weights' | 'share'>(
    key: K,
    value: NodeDef[K] | undefined,
  ) => onChange(setNodeState(spec, name, key, value))
  const wholeShare = typeof node.share === 'string'
  // 今の定義の印 (重みの報告を出すかの判定)
  const stamp = JSON.stringify(spec)
  const leftover = (['train', 'eval_mode', 'zero_init', 'weights'] as const).filter(
    (key) => key in node,
  )
  return (
    <fieldset className="space-y-2" aria-label="パラメータの状態">
      <legend className="mb-1 text-xs font-semibold">パラメータの状態</legend>
      <ShareField
        spec={spec}
        name={name}
        node={node}
        onChange={(value) => {
          if (typeof value !== 'string') return set('share', value)
          // ノード全体の共有は, 元のノードの引数に合わせ, 状態の項目を外す
          // (元のノードに従うため. 1 回の変更にまとめる)
          const next = setNodeStates(spec, name, {
            share: value,
            train: undefined,
            eval_mode: undefined,
            zero_init: undefined,
            weights: undefined,
          })
          // 元のノードに引数が無ければ, このノードの引数も外す
          const params = spec.nodes[value]?.params
          const { params: _own, ...rest } = next.nodes[name]!
          void _own
          onChange({
            ...next,
            nodes: {
              ...next.nodes,
              [name]: params === undefined ? rest : { ...rest, params: { ...params } },
            },
          })
        }}
      />
      {wholeShare ? (
        <>
          <p className="text-[11px] text-ink-3">
            ノード全体を共有するので, 学習するか, 初期値, 外部の重みは共有の元のノード (
            <span className="font-mono">{String(node.share)}</span>) に従います.
          </p>
          {/* 手で書いた定義などで状態の項目が残る場合は, 外せるように */}
          {leftover.length > 0 && (
            <div className="space-y-1 text-[11px] text-critical-text">
              <p>ノード全体を共有するノードには書けない項目 ({leftover.join(', ')}) があります.</p>
              <Button
                variant="secondary"
                icon={Trash2}
                onClick={() =>
                  onChange(
                    setNodeStates(
                      spec,
                      name,
                      Object.fromEntries(leftover.map((key) => [key, undefined])),
                    ),
                  )
                }
              >
                書けない項目を外す
              </Button>
            </div>
          )}
        </>
      ) : (
        <>
          <CheckField
            label="学習する"
            checked={node.train !== false}
            hint={
              node.share === undefined
                ? '外すと凍結 (勾配が来ず, 更新されない)'
                : '外すと凍結. 共有するパラメータは共有の元のノードの設定に従います'
            }
            onChange={(checked) => set('train', checked ? undefined : false)}
          />
          <CheckField
            label="評価の形のまま (eval_mode)"
            checked={node.eval_mode === true}
            hint="BatchNorm の統計と Dropout も止める. 凍結とは別 (既定は外)"
            onChange={(checked) => set('eval_mode', checked ? true : undefined)}
          />
          <ZeroInitField node={node} onChange={(value) => set('zero_init', value)} />
          <WeightsField
            weights={node.weights}
            onChange={(value) => set('weights', value)}
            onRead={onReadWeights}
            stamp={stamp}
          />
        </>
      )}
    </fieldset>
  )
}

/** 零初期化の欄 (しない, 全て, 名前を選ぶ) */
function ZeroInitField({
  node,
  onChange,
}: {
  node: NodeDef
  onChange: (value: NodeDef['zero_init']) => void
}) {
  const zero = node.zero_init
  const mode = zero === true ? 'all' : Array.isArray(zero) ? 'names' : 'none'
  return (
    <div className="space-y-1">
      <SelectField
        label="零初期化 (zero_init)"
        value={mode}
        options={[
          ['none', 'しない'],
          ['all', '全てのパラメータを 0 に'],
          ['names', '名前を選んで 0 に'],
        ]}
        onChange={(value) =>
          onChange(value === 'all' ? true : value === 'names' ? ['weight'] : undefined)
        }
      />
      {Array.isArray(zero) && (
        <TextField
          label="0 にするパラメータの名前 (カンマで区切る)"
          value={zero.join(', ')}
          placeholder="weight, bias"
          check={(text) => (parseNames(text).length > 0 ? null : '名前を 1 つ以上書いてください')}
          onCommit={(text) => onChange(parseNames(text))}
        />
      )}
    </div>
  )
}

/** 共有の欄 (しない, ノード全体, パラメータごと) */
function ShareField({
  spec,
  name,
  node,
  onChange,
}: {
  spec: Spec
  name: string
  node: NodeDef
  onChange: (value: NodeDef['share']) => void
}) {
  const share = node.share
  const mode = share === undefined ? 'none' : typeof share === 'string' ? 'node' : 'params'
  // ノード全体を共有できるのは, 同じ部品で自分も共有していないノード
  // (このノードを元に共有しているノードがあれば, 連鎖になるので選べない)
  const candidates = Object.entries(spec.nodes)
    .filter(([key, other]) => key !== name && other.unit === node.unit && other.share === undefined)
    .map(([key]) => key)
  const sharedBy = Object.values(spec.nodes).some((other) =>
    shareSources(other.share).includes(name),
  )
  return (
    <div className="space-y-1">
      <SelectField
        label="重みの共有 (share)"
        value={mode}
        options={[
          ['none', 'しない'],
          [
            'node',
            sharedBy
              ? 'ノード全体 (このノードを共有の元にしているノードがあるので選べません)'
              : candidates.length === 0
                ? 'ノード全体 (同じ部品のノードがありません)'
                : 'ノード全体 (同じ部品. 引数も元に合わせる)',
            mode !== 'node' && (sharedBy || candidates.length === 0),
          ],
          ['params', 'パラメータごと'],
        ]}
        onChange={(value) =>
          onChange(
            value === 'node'
              ? (candidates[0] ?? '')
              : value === 'params'
                ? { weight: `${Object.keys(spec.nodes).find((key) => key !== name) ?? ''}.weight` }
                : undefined,
          )
        }
      />
      {typeof share === 'string' && (
        <SelectField
          label="共有の元のノード"
          value={share}
          options={[
            ...(candidates.includes(share) ? [] : [[share, share || '(選ぶ)'] as [string, string]]),
            ...candidates.map((key) => [key, key] as [string, string]),
          ]}
          onChange={(value) => onChange(value)}
        />
      )}
      {share !== undefined && typeof share !== 'string' && (
        <ShareLines share={share} onChange={onChange} />
      )}
    </div>
  )
}

/** パラメータごとの共有の行 (1 行に「名前 = ノード.名前」) */
function ShareLines({
  share,
  onChange,
}: {
  share: Record<string, string>
  onChange: (value: Record<string, string>) => void
}) {
  const id = useId()
  const text = formatShareLines(share)
  const [error, setError] = useState<string | null>(null)
  // 欄の外で値が変わったら (元に戻すなど) 前の誤りを消す
  const [seen, setSeen] = useState(text)
  if (seen !== text) {
    setSeen(text)
    setError(null)
  }
  const commit = (value: string) => {
    const parsed = parseShareLines(value)
    if (typeof parsed === 'string') return setError(parsed)
    setError(null)
    if (formatShareLines(parsed) !== text) onChange(parsed)
  }
  return (
    <div>
      <label htmlFor={id} className={LABEL}>
        共有するパラメータ (1 行に「このノードの名前 = 元のノード.名前」. 形は同じにする)
      </label>
      <textarea
        id={id}
        key={text}
        defaultValue={text}
        rows={2}
        onBlur={(event) => commit(event.target.value)}
        aria-invalid={error !== null}
        className={clsx(INPUT, 'h-auto py-1')}
      />
      {error && <p className="text-[11px] text-critical-text">{error}</p>}
    </div>
  )
}

/** 外部の重みの欄 (出どころ, 入れ物のキー, キーの付け替え, strict, 読んで確かめる) */
function WeightsField({
  weights,
  onChange,
  onRead,
  stamp,
}: {
  weights: WeightsDef | undefined
  onChange: (value: WeightsDef | undefined) => void
  onRead?: () => Promise<WeightsResult>
  /** 今の定義の印 (読んだ後に定義が変わったら, 前の報告を出さない) */
  stamp: string
}) {
  const source = weightsSource(weights)
  const update = (patch: Partial<WeightsDef>) => {
    const next: WeightsDef = { ...weights, ...patch }
    for (const [key, value] of Object.entries(patch)) {
      if (value === undefined || value === '') delete next[key as keyof WeightsDef]
    }
    onChange(next)
  }
  return (
    <div className="space-y-1">
      <SelectField<WeightsSource>
        label="外部の重み (weights)"
        value={source}
        options={[
          ['none', '読まない'],
          ['path', 'ファイル (.pt, .safetensors)'],
          ['hub', 'Hugging Face の置き場'],
          ['unit', '学習済みの部品'],
        ]}
        onChange={(value) => onChange(switchWeightsSource(weights, value))}
      />
      {weights && (
        <div className="space-y-1 rounded-md border border-line p-2">
          {source === 'path' && (
            <TextField
              label="ファイルのパス (相対のパスは定義の置き場から)"
              value={weights.path ?? ''}
              placeholder="weights/base.pt"
              onCommit={(text) => onChange({ ...weights, path: text })}
            />
          )}
          {source === 'hub' && (
            <>
              <TextField
                label="置き場の名前"
                value={weights.hub ?? ''}
                placeholder="組織/名前"
                onCommit={(text) => onChange({ ...weights, hub: text })}
              />
              <TextField
                label="ファイル"
                value={weights.file ?? ''}
                placeholder="model.safetensors"
                onCommit={(text) => onChange({ ...weights, file: text })}
              />
              <TextField
                label="版 (既定 main)"
                value={weights.revision ?? ''}
                onCommit={(text) => update({ revision: text })}
              />
            </>
          )}
          {source === 'unit' && (
            <>
              <TextField
                label="部品の名前"
                value={weights.unit ?? ''}
                placeholder="timm.Model"
                onCommit={(text) => onChange({ ...weights, unit: text })}
              />
              <TextField
                label="部品の引数 (JSON)"
                value={JSON.stringify(weights.params ?? {})}
                check={(text) => {
                  try {
                    const value: unknown = JSON.parse(text)
                    return value && typeof value === 'object' && !Array.isArray(value)
                      ? null
                      : '{"model_name": "resnet18", "pretrained": true} の形にしてください'
                  } catch {
                    return 'JSON として読めません'
                  }
                }}
                onCommit={(text) =>
                  onChange({ ...weights, params: JSON.parse(text) as Record<string, unknown> })
                }
              />
            </>
          )}
          <TextField
            label="入れ物のキー (state_key. 省略可)"
            value={weights.state_key ?? ''}
            placeholder="state_dict"
            onCommit={(text) => update({ state_key: text })}
          />
          <RemapRows
            rules={weights.remap ?? []}
            onChange={(rules) => update({ remap: rules.length > 0 ? rules : undefined })}
          />
          <CheckField
            label="合わない重みがあれば止める (strict. 既定で入)"
            checked={weights.strict !== false}
            hint="外すと, 形の違う重みと新しい重みは今の値のまま読む (頭を差し替える場合など)"
            onChange={(checked) => update({ strict: checked ? undefined : false })}
          />
          {onRead && <ReadWeights onRead={onRead} stamp={stamp} />}
        </div>
      )}
    </div>
  )
}

/** キーの付け替えの並び (1 行ごとに前置きか正規表現) */
function RemapRows({
  rules,
  onChange,
}: {
  rules: RemapRule[]
  onChange: (rules: RemapRule[]) => void
}) {
  const replace = (index: number, rule: RemapRule) =>
    onChange(rules.map((item, at) => (at === index ? rule : item)))
  return (
    <div className="space-y-1">
      <div className={LABEL}>
        キーの付け替え (上から見て最初に当たった 1 行で. 正規表現は番号を付け替える場合だけ)
      </div>
      {rules.map((rule, index) => (
        <div
          key={index}
          role="group"
          aria-label={`付け替え ${index + 1}`}
          className="space-y-1 rounded-md bg-surface-2 p-1"
        >
          <div className="flex items-end gap-1">
            <div className="flex-1">
              <SelectField
                label="種類"
                value={ruleKind(rule)}
                options={[
                  ['prefix', '前置き'],
                  ['pattern', '正規表現'],
                ]}
                onChange={(kind) => replace(index, switchRuleKind(rule, kind))}
              />
            </div>
            <Button
              variant="secondary"
              icon={Trash2}
              onClick={() => onChange(rules.filter((_, at) => at !== index))}
              title="この付け替えを消す"
            >
              消す
            </Button>
          </div>
          {ruleKind(rule) === 'prefix' ? (
            <TextField
              label="前置き (ファイルのキー. 空なら全てのキーに当たる)"
              value={rule.from ?? ''}
              placeholder="model.blocks."
              onCommit={(text) => replace(index, { ...rule, from: text })}
            />
          ) : (
            <TextField
              label="正規表現 (ファイルのキー)"
              value={rule.pattern ?? ''}
              placeholder="^layer\.(\d+)\."
              onCommit={(text) => replace(index, { ...rule, pattern: text })}
            />
          )}
          <TextField
            label="置き換え (ノードのキー)"
            value={rule.to}
            placeholder={ruleKind(rule) === 'prefix' ? 'nodes.' : 'nodes.block\\1.'}
            onCommit={(text) => replace(index, { ...rule, to: text })}
          />
          {ruleKind(rule) === 'pattern' && (
            <TextField
              label="番号に足す数 (offset. 省略可)"
              value={rule.offset === undefined ? '' : String(rule.offset)}
              placeholder="-6"
              check={(text) => (text === '' || /^-?\d+$/.test(text) ? null : '整数にしてください')}
              onCommit={(text) => {
                const { offset: _offset, ...rest } = rule
                void _offset
                replace(index, text === '' ? rest : { ...rest, offset: Number(text) })
              }}
            />
          )}
        </div>
      ))}
      <Button
        variant="secondary"
        icon={Plus}
        onClick={() => onChange([...rules, { from: '', to: '' }])}
      >
        付け替えを足す
      </Button>
    </div>
  )
}

/** 外部の重みを実際に読み, ノードごとの数を出す */
function ReadWeights({ onRead, stamp }: { onRead: () => Promise<WeightsResult>; stamp: string }) {
  const read = useMutation({ mutationFn: onRead })
  // 読んだときの定義の印 (定義が変われば前の報告は今の定義のものではない)
  const [readAt, setReadAt] = useState<string | null>(null)
  const current = readAt === stamp
  return (
    <div className="space-y-1">
      <Button
        variant="secondary"
        icon={Download}
        onClick={() => {
          setReadAt(stamp)
          read.mutate()
        }}
        disabled={read.isPending}
        title="定義全体を組み立てて外部の重みを読む (ファイルとネットワークを使う)"
      >
        {read.isPending ? '重みを読んでいます' : '重みを読んで確かめる'}
      </Button>
      {current && read.isError && (
        <p role="alert" className="text-xs text-critical-text">
          読めませんでした: {read.error.message}
        </p>
      )}
      {current && read.data && !read.data.ok && <WeightsError result={read.data} />}
      {current && read.data?.ok && (
        <section role="status" aria-label="重みの読み込みの報告" className="space-y-2">
          {read.data.reports.length === 0 && (
            <p className="text-xs text-ink-3">外部の重みを読むノードがありません.</p>
          )}
          {read.data.reports.map((report) => (
            <ReportCard key={report.node} report={report} />
          ))}
        </section>
      )}
    </div>
  )
}

/** 重みを読めなかった誤りの文. 申請制の重みは, 配布元のページで申請する案内も出す */
export function WeightsError({ result }: { result: WeightsResult }) {
  return (
    <div role="alert" className="space-y-1 text-xs text-critical-text">
      <p className="[overflow-wrap:anywhere]">{result.error}</p>
      {result.gated?.url && (
        // 申請制の重みは, 申請と条件を確かめる配布元のページへ
        <a
          href={result.gated.url}
          target="_blank"
          rel="noreferrer"
          className="inline-flex items-center gap-1 text-accent-text underline"
        >
          配布元のページで申請する
          <ExternalLink className="size-3" aria-hidden />
          <span className="sr-only">(新しいタブで開く)</span>
        </a>
      )}
    </div>
  )
}

/** 1 つのノードの読み込みの報告 (数, 出どころ, キーの一部, ライセンス) */
function ReportCard({ report }: { report: WeightsReport }) {
  const counts: [string, number][] = [
    ['読めた', report.loaded],
    ['形の違う', report.mismatched],
    ['新しい', report.new],
    ['使わない', report.unused],
  ]
  return (
    <div className="space-y-1 rounded-md bg-surface-2 p-2 text-xs">
      <p>
        <span className="font-mono font-semibold">{report.node}</span>{' '}
        <span className="text-ink-3">{report.source}</span>
      </p>
      <dl className="grid grid-cols-4 gap-1 text-center">
        {counts.map(([label, count]) => (
          <div key={label}>
            <dt className="text-[11px] text-ink-3">{label}</dt>
            <dd className="font-mono" data-count={label}>
              {count}
            </dd>
          </div>
        ))}
      </dl>
      <KeyList
        label="形の違う重み (キー: ファイル → ノード)"
        keys={(report.mismatched_keys ?? []).map((item) =>
          Array.isArray(item)
            ? `${String(item[0])}: ${JSON.stringify(item[1])} → ${JSON.stringify(item[2])}`
            : String(item),
        )}
        total={report.mismatched}
      />
      <KeyList
        label="新しい重み (ファイルに無い)"
        keys={report.new_keys ?? []}
        total={report.new}
      />
      <KeyList
        label="使わない重み (付け替えた後のキー)"
        keys={report.unused_keys ?? []}
        total={report.unused}
      />
      {(report.warnings ?? []).map((message) => (
        <p key={message} className="text-warning-text [overflow-wrap:anywhere]">
          {message}
        </p>
      ))}
      {report.license && (
        <LicenseCard info={{ ...report.license, unit: report.license.unit ?? '' }} />
      )}
    </div>
  )
}

/** 報告のキーの一部 (畳んで出す) */
function KeyList({ label, keys, total }: { label: string; keys: string[]; total: number }) {
  if (keys.length === 0) return null
  return (
    <details>
      <summary className="cursor-pointer text-[11px] text-ink-3">
        {label} ({keys.length < total ? `${keys.length} / ${total}` : total})
      </summary>
      <ul className="font-mono text-[11px] [overflow-wrap:anywhere]">
        {keys.map((key) => (
          <li key={key}>{key}</li>
        ))}
      </ul>
    </details>
  )
}
