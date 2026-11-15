// 確認の誤りと注意の, 画面に出す形

import type { Issue } from '../api/client'

/**
 * 誤りの場所を画面向けにする (引数とつなぎ先は名前だけにし, 定義全体の項目の
 * 場所は文に含まれるので出さない)
 */
export function friendlyLocation(issue: Issue): string | null {
  const location = issue.location ?? ''
  if (!location || ['nodes', 'outputs', 'inputs'].includes(location)) return null
  return location.replace(/^(params|forward_params|inputs)\./, '')
}

/** 画面の操作で直す誤りは, 画面の言葉の文にする (定義の項目の名前を出さない) */
export function friendlyMessage(issue: Issue): string {
  if (issue.code === 'no_nodes') return '部品を 1 つ以上置いてください'
  if (issue.code === 'no_outputs') return '出力を 1 つ以上加えてください (上の「出力」)'
  if (issue.code === 'no_inputs') return '入力を 1 つ以上加えてください (上の「入力」)'
  return issue.message
}
