// 部品の一覧のまとまり (出どころごと. 起動の設定のモジュールの自作の部品はモジュールごと)

import type { UnitSpec } from './graph'

// 出どころの表示名と並び順
export const SOURCES: Record<string, string> = {
  subgraph: '部分モデル',
  ml_modelcontrol: '演算',
  'torch.nn': 'PyTorch',
  torch_geometric: 'PyTorch Geometric',
  transformers: 'Hugging Face',
  torchvision: 'torchvision',
  timm: 'timm',
  torchaudio: 'torchaudio',
  einops: 'einops',
  kornia: 'Kornia',
  monai: 'MONAI',
  segmentation_models_pytorch: 'smp (segmentation_models_pytorch)',
  custom: '自作',
}

/** 起動の設定のモジュールのまとまりの鍵の頭 */
const MODULE_PREFIX = 'custom:'

/** 部品のまとまりの鍵 (自作の部品はモジュールごと) */
export function groupKey(unit: UnitSpec): string {
  if (unit.source === 'custom' && unit.module) return `${MODULE_PREFIX}${unit.module}`
  return unit.source
}

/** まとまりの表示名 */
export function groupLabel(key: string): string {
  if (key.startsWith(MODULE_PREFIX)) return `自作 (${key.slice(MODULE_PREFIX.length)})`
  return SOURCES[key] ?? key
}

/** 起動の設定のモジュールのまとまりか (初めから開く) */
export function isModuleGroup(key: string): boolean {
  return key.startsWith(MODULE_PREFIX)
}

/**
 * 部品をまとまりごとに並べる (出どころの順. モジュールのまとまりは組み込みの部品の後に
 * モジュールの名前の順. 知らない出どころは最後)
 */
export function groupUnits(units: UnitSpec[]): [string, UnitSpec[]][] {
  const order = Object.keys(SOURCES)
  const groups = new Map<string, UnitSpec[]>()
  for (const unit of units) {
    const key = groupKey(unit)
    const list = groups.get(key)
    if (list) list.push(unit)
    else groups.set(key, [unit])
  }
  const rank = (key: string) => {
    if (isModuleGroup(key)) return order.length
    const index = order.indexOf(key)
    return index < 0 ? order.length + 1 : index
  }
  return [...groups.entries()].sort(
    ([a], [b]) => rank(a) - rank(b) || (isModuleGroup(a) ? a.localeCompare(b) : 0),
  )
}
