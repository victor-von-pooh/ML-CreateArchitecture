import { Handle, Position, type Node, type NodeProps } from '@xyflow/react'
import clsx from 'clsx'
import { AlertTriangle, ArrowRightFromLine, ArrowRightToLine } from 'lucide-react'

import type { NodeResult } from '../../api/client'
import type { InputDef, UnitSpec } from '../../lib/graph'
import { formatShape } from '../../lib/format'

/** 部品のノードの表示に使う値 */
export type UnitNodeData = {
  name: string
  unit: UnitSpec | undefined
  unitName: string
  result: NodeResult | undefined
  errors: number
}

/** 入力のノードの表示に使う値 */
export type InputNodeData = { name: string; input: InputDef; errors: number }

/** 出力のノードの表示に使う値 */
export type OutputNodeData = { name: string; shape: string | null; connected: boolean }

export type UnitFlowNode = Node<UnitNodeData, 'unit'>
export type InputFlowNode = Node<InputNodeData, 'modelInput'>
export type OutputFlowNode = Node<OutputNodeData, 'modelOutput'>

// 口の行の高さ (ハンドルの位置をそろえる)
const ROW = 22

/** ノードの枠 (選択と誤りで色を変える) */
function frame(selected: boolean, errors: number): string {
  return clsx(
    'rounded-lg border bg-surface text-ink shadow-sm',
    selected ? 'border-accent ring-2 ring-accent/30' : 'border-line',
    errors > 0 && !selected && 'border-critical',
  )
}

/** 部品のノード. 左に入力の口, 右に出力の口を並べる */
export function UnitNode({ data, selected }: NodeProps<UnitFlowNode>) {
  const inputs = data.unit?.inputs ?? []
  const outputs = data.unit?.outputs ?? [{ name: 'out', description: '' }]
  const rows = Math.max(inputs.length, outputs.length, 1)
  const status = data.result?.status
  return (
    <div className={frame(selected, data.errors)} style={{ minWidth: 190 }}>
      <div className="flex items-center justify-between gap-2 border-b border-line px-3 py-1.5">
        <div className="min-w-0">
          <div className="truncate font-mono text-xs font-semibold">{data.name}</div>
          <div className="truncate text-[11px] text-ink-3">{data.unitName}</div>
        </div>
        {(data.errors > 0 || status === 'error') && (
          <span className="flex items-center gap-0.5 text-[11px] text-critical-text">
            <AlertTriangle className="size-3.5" aria-hidden />
            {data.errors > 0 ? data.errors : '!'}
            <span className="sr-only">件の誤り</span>
          </span>
        )}
      </div>
      <div className="relative" style={{ height: rows * ROW + 8 }}>
        {inputs.map((port, index) => (
          <div
            key={port.name}
            className="absolute left-2 font-mono text-[10px] text-ink-2"
            style={{ top: 4 + index * ROW }}
          >
            <Handle
              type="target"
              position={Position.Left}
              id={`in:${port.name}`}
              style={{ top: 9, left: -13 }}
              className={clsx(port.optional && '!bg-surface-2')}
            />
            {port.name}
            {port.variadic ? ' …' : port.optional ? ' ?' : ''}
          </div>
        ))}
        {outputs.map((port, index) => {
          const shape = data.result?.outputs?.[port.name]
          return (
            <div
              key={port.name}
              className="absolute right-2 text-right font-mono text-[10px]"
              style={{ top: 4 + index * ROW }}
            >
              <span className="text-ink-3">{shape ? formatShape(shape) : ''}</span>{' '}
              <span className="text-ink-2">{port.name}</span>
              <Handle
                type="source"
                position={Position.Right}
                id={`out:${port.name}`}
                style={{ top: 9, right: -13 }}
              />
            </div>
          )
        })}
      </div>
      {status === 'skipped' && (
        <div className="border-t border-line px-3 py-1 text-[10px] text-ink-3">
          前のノードの誤りで計算できません
        </div>
      )}
    </div>
  )
}

/** モデルの入力のノード */
export function InputNode({ data, selected }: NodeProps<InputFlowNode>) {
  return (
    <div className={frame(selected, data.errors)} style={{ minWidth: 140 }}>
      <div className="flex items-center gap-1.5 px-3 py-2">
        <ArrowRightToLine className="size-3.5 text-accent-text" aria-hidden />
        <div>
          <div className="font-mono text-xs font-semibold">{data.name}</div>
          <div className="font-mono text-[10px] text-ink-3">
            [{(data.input.shape ?? []).join(', ')}] {data.input.dtype ?? 'float32'}
          </div>
        </div>
      </div>
      <Handle type="source" position={Position.Right} id="out:out" />
    </div>
  )
}

/** モデルの出力のノード */
export function OutputNode({ data, selected }: NodeProps<OutputFlowNode>) {
  return (
    <div className={frame(selected, data.connected ? 0 : 1)} style={{ minWidth: 140 }}>
      <Handle type="target" position={Position.Left} id="in:in" />
      <div className="flex items-center gap-1.5 px-3 py-2">
        <ArrowRightFromLine className="size-3.5 text-accent-text" aria-hidden />
        <div>
          <div className="font-mono text-xs font-semibold">{data.name}</div>
          <div className="font-mono text-[10px] text-ink-3">
            {data.connected ? (data.shape ?? '') : '未接続'}
          </div>
        </div>
      </div>
    </div>
  )
}
