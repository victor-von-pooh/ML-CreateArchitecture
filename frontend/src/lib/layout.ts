import { specWires, type Spec, type Ui, type UnitSpec } from './graph'

// 段 (つながりの深さ) の間の横の間隔と, 同じ段の中の縦の間隔
const COLUMN = 260
const GAP = 40

// 1 行の段の数の上限 (超える場合は折り返し, 行の段の数を揃える) と, 行の間
const WRAP_AFTER = 8
const ROW_GAP = 120

// 段を飛び越すつなぎのために, またぐ段のノードを下げる量
const LANE = 110

// ノードの高さの見積もり (見出しと, 口の行ごとの高さ)
const HEADER = 60
const PORT_ROW = 22

/** 位置 */
export interface Point {
  x: number
  y: number
}

/** 出力の名前 (つないだものと, 位置だけを持つ未接続のもの) */
function outputNames(spec: Spec): string[] {
  return [...new Set([...Object.keys(spec.outputs), ...Object.keys(spec.ui?.outputs ?? {})])]
}

/** ノードの高さの見積もり (入力と出力の口の多い方の行の数から) */
function heightOf(id: string, spec: Spec, units: Map<string, UnitSpec>): number {
  const node = spec.nodes[id]
  if (!node) return HEADER
  const unit = units.get(node.unit)
  const rows = Math.max(unit?.inputs.length ?? 1, unit?.outputs.length ?? 1, 1)
  return HEADER + rows * PORT_ROW
}

/**
 * つながりの深さで左から右に段を分けた, 全てのもの (入力, ノード, 出力) の位置
 *
 * 入力は最初の段, ノードはつなぎ元の最も深い段の次, 出力は最後の段.
 * 同じ段の中は定義に書いた順に上から並べる. 段が多い場合は, 段を
 * 折り返して下の行に続ける
 */
export function layoutPositions(spec: Spec, units: Map<string, UnitSpec>): Map<string, Point> {
  // ノードごとのつなぎ元
  const sources = new Map<string, string[]>()
  for (const wire of specWires(spec, units)) {
    if (!(wire.target in spec.nodes)) continue
    sources.set(wire.target, [...(sources.get(wire.target) ?? []), wire.source])
  }

  // 深さ (循環している場合は, 回っている途中のつなぎ元を数えない)
  const depth = new Map<string, number>()
  const visiting = new Set<string>()
  const depthOf = (id: string): number => {
    if (id in spec.inputs) return 0
    if (!(id in spec.nodes)) return 0
    const known = depth.get(id)
    if (known !== undefined) return known
    if (visiting.has(id)) return 0
    visiting.add(id)
    const value = 1 + Math.max(0, ...(sources.get(id) ?? []).map(depthOf))
    visiting.delete(id)
    depth.set(id, value)
    return value
  }

  // 段ごとに, 定義に書いた順で集める
  const columns = new Map<number, string[]>()
  const place = (id: string, column: number) => {
    columns.set(column, [...(columns.get(column) ?? []), id])
  }
  Object.keys(spec.inputs).forEach((id) => place(id, 0))
  const nodeDepths = Object.keys(spec.nodes).map((id) => [id, depthOf(id)] as const)
  nodeDepths.forEach(([id, value]) => place(id, value))
  const last = Math.max(0, ...nodeDepths.map(([, value]) => value)) + 1
  outputNames(spec).forEach((name) => place(`output:${name}`, last))

  // 段を飛び越すつなぎ (残差など) がまたぐ段は, 線がノードの裏を通らない
  // ように, ノードを 1 つ分下げて上を空ける
  const columnOf = (id: string) =>
    id in spec.inputs ? 0 : id.startsWith('output:') ? last : (depth.get(id) ?? 0)
  const lifted = new Set<number>()
  for (const wire of specWires(spec, units)) {
    const from = columnOf(wire.source)
    const to = columnOf(wire.target)
    for (let column = from + 1; column < to; column += 1) lifted.add(column)
  }
  const offset = (column: number) => (lifted.has(column) ? LANE : 0)

  // 段が多い場合は, 全体が横に長くなりすぎないように折り返して行に並べる
  // 飛び越すつなぎ (U-Net の skip など) がある図は, 折り返すと線が行を
  // またいで読めないので折り返さない
  const count = last + 1
  const rows = lifted.size > 0 ? 1 : Math.ceil(count / WRAP_AFTER)
  const perRow = Math.ceil(count / rows)
  const heights = new Map<number, number>()
  for (const [column, ids] of columns) {
    const total = offset(column) + ids.reduce((sum, id) => sum + heightOf(id, spec, units) + GAP, 0)
    const row = Math.floor(column / perRow)
    heights.set(row, Math.max(heights.get(row) ?? 0, total))
  }
  const top = (row: number) => {
    let y = 60
    for (let index = 0; index < row; index += 1) y += (heights.get(index) ?? 0) + ROW_GAP
    return y
  }

  // 段の中は見積もった高さで上から詰める
  const positions = new Map<string, Point>()
  for (const [column, ids] of columns) {
    const row = Math.floor(column / perRow)
    let y = top(row) + offset(column)
    for (const id of ids) {
      positions.set(id, { x: 40 + (column % perRow) * COLUMN, y })
      y += heightOf(id, spec, units) + GAP
    }
  }
  return positions
}

/** 全てのものを段に分けて並べ直した定義 (位置は ui に書く) */
export function arrangeSpec(spec: Spec, units: Map<string, UnitSpec>): Spec {
  const positions = layoutPositions(spec, units)
  const at = (id: string): Point => positions.get(id) ?? { x: 40, y: 60 }
  const inputs = Object.fromEntries(
    Object.entries(spec.inputs).map(([name, input]) => [
      name,
      { ...input, ui: { ...input.ui, ...at(name) } },
    ]),
  )
  const nodes = Object.fromEntries(
    Object.entries(spec.nodes).map(([name, node]) => [
      name,
      { ...node, ui: { ...node.ui, ...at(name) } },
    ]),
  )
  const outputs: Record<string, Ui> = Object.fromEntries(
    outputNames(spec).map((name) => [
      name,
      { ...spec.ui?.outputs?.[name], ...at(`output:${name}`) },
    ]),
  )
  return { ...spec, inputs, nodes, ui: { ...spec.ui, outputs } }
}

/** 位置の無いもの (入力, ノード, 出力) にだけ, 段に分けた位置を書いた定義 */
export function placeMissing(spec: Spec, units: Map<string, UnitSpec>): Spec {
  const arranged = arrangeSpec(spec, units)
  const has = (ui: { x?: number; y?: number } | undefined) =>
    ui?.x !== undefined && ui?.y !== undefined
  const inputs = Object.fromEntries(
    Object.entries(spec.inputs).map(([name, input]) => [
      name,
      has(input.ui) ? input : arranged.inputs[name]!,
    ]),
  )
  const nodes = Object.fromEntries(
    Object.entries(spec.nodes).map(([name, node]) => [
      name,
      has(node.ui) ? node : arranged.nodes[name]!,
    ]),
  )
  const outputs = Object.fromEntries(
    Object.entries(arranged.ui?.outputs ?? {}).map(([name, ui]) => [
      name,
      has(spec.ui?.outputs?.[name]) ? spec.ui!.outputs![name]! : ui,
    ]),
  )
  return { ...spec, inputs, nodes, ui: { ...spec.ui, outputs } }
}

/** 置いてあるものの枠 */
export interface Rect {
  x: number
  y: number
  width: number
  height: number
}

/**
 * 望む位置の近くで, 置いてあるものと重ならない新しいノードの位置
 *
 * 望む位置から, 縦と横に少しずつ離れた所を近い順に試す. 見えている範囲
 * (bounds) の中に空きが無い場合は望む位置 (重なる)
 */
export function freeSpot(
  rects: Rect[],
  wanted: Point,
  size = { width: 200, height: 100 },
  bounds?: Rect,
): Point {
  const margin = 20
  const overlaps = (point: Point) =>
    rects.some(
      (rect) =>
        point.x < rect.x + rect.width + margin &&
        point.x + size.width + margin > rect.x &&
        point.y < rect.y + rect.height + margin &&
        point.y + size.height + margin > rect.y,
    )
  const candidates: Point[] = []
  for (let row = -6; row <= 6; row += 1) {
    for (let column = -4; column <= 4; column += 1) {
      candidates.push({ x: wanted.x + column * 240, y: wanted.y + row * 130 })
    }
  }
  // 見えている範囲が分かる場合は, その中に収まる所だけを試す
  const inside = (point: Point) =>
    !bounds ||
    (point.x >= bounds.x &&
      point.y >= bounds.y &&
      point.x + size.width <= bounds.x + bounds.width &&
      point.y + size.height <= bounds.y + bounds.height)
  candidates.sort(
    (a, b) =>
      Math.hypot(a.x - wanted.x, a.y - wanted.y) - Math.hypot(b.x - wanted.x, b.y - wanted.y),
  )
  return candidates.find((point) => inside(point) && !overlaps(point)) ?? wanted
}

/** 部品のノードの大きさの見積もり (口の数から. 新しく置く所を探すため) */
export function unitSize(unit: UnitSpec | undefined): { width: number; height: number } {
  const rows = Math.max(unit?.inputs.length ?? 1, unit?.outputs.length ?? 1, 1)
  return { width: 220, height: HEADER + rows * PORT_ROW }
}
