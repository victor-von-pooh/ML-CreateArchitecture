import type { Edge, Node } from '@xyflow/react'

import type { CheckResult } from '../api/client'
import { formatShape } from './format'
import { specWires, type Spec, type UnitSpec } from './graph'

/** 位置の無いノードの置き場所 (横に並べる) */
function fallbackPosition(index: number, row: number): { x: number; y: number } {
  return { x: 40 + index * 240, y: 60 + row * 200 }
}

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

  const nodes: Node[] = []
  Object.entries(spec.inputs).forEach(([name, input], index) => {
    nodes.push({
      id: name,
      type: 'modelInput',
      position: { x: input.ui?.x ?? 40, y: input.ui?.y ?? 60 + index * 120 },
      data: { name, input, errors: errors.get(`inputs.${name}`) ?? 0 },
      selected: selected === name,
    })
  })
  Object.entries(spec.nodes).forEach(([name, node], index) => {
    const fallback = fallbackPosition(index + 1, 0)
    nodes.push({
      id: name,
      type: 'unit',
      position: { x: node.ui?.x ?? fallback.x, y: node.ui?.y ?? fallback.y },
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
  outputNames.forEach((name, index) => {
    const ui = spec.ui?.outputs?.[name]
    const fallback = fallbackPosition(Object.keys(spec.nodes).length + 1, index)
    const shape = check?.outputs?.[name]
    nodes.push({
      id: `output:${name}`,
      type: 'modelOutput',
      position: { x: ui?.x ?? fallback.x, y: ui?.y ?? fallback.y },
      data: { name, shape: shape ? formatShape(shape) : null, connected: name in spec.outputs },
      selected: selected === `output:${name}`,
    })
  })
  return nodes
}

/** 定義から, キャンバスの線の一覧を作る */
export function flowEdges(spec: Spec, units: Map<string, UnitSpec>): Edge[] {
  return specWires(spec, units).map((wire) => ({
    id: wire.id,
    source: wire.source,
    sourceHandle: `out:${wire.sourcePort}`,
    target: wire.target,
    targetHandle: `in:${wire.targetPort}`,
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
