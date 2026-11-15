import type { Edge, Node } from '@xyflow/react'

import type { CheckResult } from '../api/client'
import { formatShape } from './format'
import { specWires, type Spec, type UnitSpec } from './graph'
import { layoutPositions } from './layout'

// 部品の一覧からキャンバスにドラッグする部品の名前のデータの種類
export const UNIT_DRAG_TYPE = 'application/x-ml-unit'

/** 定義から, キャンバスのノードの一覧を作る */
export function flowNodes(
  spec: Spec,
  units: Map<string, UnitSpec>,
  check: CheckResult | undefined,
  selected: string | null,
): Node[] {
  // ノードごとの誤りの数
  const errors = new Map<string, number>()
  for (const issue of check?.issues ?? []) {
    if (issue.severity !== 'error') continue
    const key = issue.node ?? (issue.location?.split('.').slice(0, 2).join('.') || '')
    errors.set(key, (errors.get(key) ?? 0) + 1)
  }

  // 位置の無いもの (手で書いた定義など) は, つながりの深さで段に分けた位置
  const layout = layoutPositions(spec, units)
  const fallback = (id: string) => layout.get(id) ?? { x: 40, y: 60 }

  const nodes: Node[] = []
  Object.entries(spec.inputs).forEach(([name, input]) => {
    nodes.push({
      id: name,
      type: 'modelInput',
      // 読み上げの名前 (種類と形)
      ariaLabel: `入力 ${name} [${(input.shape ?? []).join(', ')}]`,
      position: { x: input.ui?.x ?? fallback(name).x, y: input.ui?.y ?? fallback(name).y },
      data: { name, input, errors: errors.get(`inputs.${name}`) ?? 0 },
      selected: selected === name,
    })
  })
  Object.entries(spec.nodes).forEach(([name, node]) => {
    nodes.push({
      id: name,
      type: 'unit',
      ariaLabel: unitLabel(name, node.unit, check?.nodes?.[name], errors.get(name) ?? 0),
      position: { x: node.ui?.x ?? fallback(name).x, y: node.ui?.y ?? fallback(name).y },
      data: {
        name,
        unit: units.get(node.unit),
        unitName: node.unit,
        result: check?.nodes?.[name],
        errors: errors.get(name) ?? 0,
      },
      selected: selected === name,
    })
  })
  const outputNames = [
    ...new Set([...Object.keys(spec.outputs), ...Object.keys(spec.ui?.outputs ?? {})]),
  ]
  outputNames.forEach((name) => {
    const ui = spec.ui?.outputs?.[name]
    const id = `output:${name}`
    const shape = check?.outputs?.[name]
    nodes.push({
      id,
      type: 'modelOutput',
      ariaLabel: `出力 ${name}${name in spec.outputs ? '' : ' (未接続)'}`,
      position: { x: ui?.x ?? fallback(id).x, y: ui?.y ?? fallback(id).y },
      data: { name, shape: shape ? formatShape(shape) : null, connected: name in spec.outputs },
      selected: selected === `output:${name}`,
    })
  })
  return nodes
}

/** 定義から, キャンバスの線の一覧を作る */
export function flowEdges(
  spec: Spec,
  units: Map<string, UnitSpec>,
  selected: string | null = null,
): Edge[] {
  return specWires(spec, units).map((wire) => ({
    id: wire.id,
    selected: wire.id === selected,
    source: wire.source,
    sourceHandle: `out:${wire.sourcePort}`,
    target: wire.target,
    targetHandle: `in:${wire.targetPort}`,
    // 読み上げの名前 (どこの口からどこの口へ)
    ariaLabel: `${wire.source}${wire.sourcePort === 'out' ? '' : ` の ${wire.sourcePort}`} から ${
      wire.target.startsWith('output:')
        ? `出力 ${wire.target.slice(7)}`
        : `${wire.target} の ${wire.targetPort}`
    } への線`,
  }))
}

/** ノードの位置を定義の ui に書く */
export function moveItem(spec: Spec, id: string, position: { x: number; y: number }): Spec {
  const ui = { x: Math.round(position.x), y: Math.round(position.y) }
  if (id.startsWith('output:')) {
    const name = id.slice('output:'.length)
    return {
      ...spec,
      ui: { ...spec.ui, outputs: { ...(spec.ui?.outputs ?? {}), [name]: ui } },
    }
  }
  if (id in spec.inputs) {
    const input = spec.inputs[id]!
    return { ...spec, inputs: { ...spec.inputs, [id]: { ...input, ui: { ...input.ui, ...ui } } } }
  }
  const node = spec.nodes[id]
  if (!node) return spec
  return { ...spec, nodes: { ...spec.nodes, [id]: { ...node, ui: { ...node.ui, ...ui } } } }
}

/** 部品のノードの読み上げの名前 (名前, 部品, 出力の形, 誤りの数) */
function unitLabel(
  name: string,
  unit: string,
  result: { status: string; outputs?: Record<string, unknown> } | undefined,
  errors: number,
): string {
  const shapes = Object.entries(result?.outputs ?? {})
    .map(([port, shape]) => `${port} ${formatShape(shape)}`)
    .join(', ')
  const state =
    errors > 0
      ? `, 誤り ${errors} 件`
      : result?.status === 'error'
        ? ', 計算の誤り'
        : result?.status === 'skipped'
          ? ', 前のノードの誤りで計算できない'
          : ''
  return `ノード ${name} (${unit})${shapes ? `, 出力 ${shapes}` : ''}${state}`
}
