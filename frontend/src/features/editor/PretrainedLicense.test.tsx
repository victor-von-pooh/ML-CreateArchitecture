import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import type { PretrainedInfo } from '../../api/client'
import { WeightsError } from './NodeState'
import { LicenseCard } from './PretrainedLicense'

// 配布元から読めたライセンスの情報 (値だけを変える)
const info = (extra: Partial<PretrainedInfo> = {}): PretrainedInfo => ({
  unit: 'hf.AutoModel',
  library: 'huggingface',
  model: 'org/model',
  hub_id: 'org/model',
  revision: 'main',
  pinned_revision: 'a'.repeat(40),
  license: 'apache-2.0',
  licenses: ['apache-2.0'],
  confirmed: true,
  status: 'commercial',
  redistributable: true,
  gated: false,
  alert: false,
  message: 'ライセンス: apache-2.0 (商用で使える)',
  source: 'Hugging Face のモデルの説明から読んだ (org/model)',
  url: 'https://huggingface.co/org/model',
  author: 'org',
  attribution: 'ライセンスの文と著作権の表示を添える',
  credit:
    'org の org/model (版 aaaaaaaaaaaa), https://huggingface.co/org/model, ライセンス apache-2.0',
  size_bytes: 440 * 1024 * 1024,
  ...extra,
})

describe('LicenseCard', () => {
  it('表示の義務, 出どころの表記, 重みの大きさを出す', () => {
    render(<LicenseCard info={info()} />)
    expect(screen.getByText('表示の義務').nextElementSibling).toHaveTextContent(
      'ライセンスの文と著作権の表示を添える',
    )
    expect(screen.getByText('出どころの表記').nextElementSibling).toHaveTextContent(
      /^org の org\/model/,
    )
    expect(screen.getByText('重みの大きさ').nextElementSibling).toHaveTextContent('440 MB')
  })

  it('値の無いものは出さない', () => {
    render(<LicenseCard info={info({ attribution: '', credit: null, size_bytes: null })} />)
    expect(screen.queryByText('表示の義務')).toBeNull()
    expect(screen.queryByText('出どころの表記')).toBeNull()
    expect(screen.queryByText('重みの大きさ')).toBeNull()
  })

  it('読めない大きさ (負の値) は出さない', () => {
    render(<LicenseCard info={info({ size_bytes: -1 })} />)
    expect(screen.queryByText('重みの大きさ')).toBeNull()
  })
})

describe('WeightsError', () => {
  it('申請制の重みは, 誤りの文と配布元のページへの案内を出す', () => {
    render(
      <WeightsError
        result={{
          ok: false,
          reports: [],
          error: 'ノード bert の部品を作れません: 申請制の重みを読めません: …',
          gated: {
            reason: 'not_granted',
            hub_id: 'org/gated',
            url: 'https://huggingface.co/org/gated',
          },
        }}
      />,
    )
    expect(screen.getByRole('alert')).toHaveTextContent(/^ノード bert の部品を作れません/)
    expect(screen.getByRole('link', { name: /配布元のページで申請する/ })).toHaveAttribute(
      'href',
      'https://huggingface.co/org/gated',
    )
  })

  // 申請制の誤りの結果 (理由と配布元のページだけを変える)
  const gatedResult = (reason: string, url: string | null) => ({
    ok: false,
    reports: [],
    error: '申請制の重みを読めません',
    gated: { reason, hub_id: 'org/gated', url },
  })

  it('リンクの文を理由ごとに変える', () => {
    const { rerender } = render(
      <WeightsError result={gatedResult('no_token', 'https://huggingface.co/org/gated')} />,
    )
    expect(screen.getByRole('link')).toHaveTextContent('配布元のページで申請する')
    rerender(
      <WeightsError result={gatedResult('invalid_token', 'https://huggingface.co/org/gated')} />,
    )
    expect(screen.getByRole('link')).toHaveTextContent('配布元のページを開く')
    expect(screen.getByRole('link')).not.toHaveTextContent('申請')
  })

  it('https で始まらない配布元のページはリンクにしない', () => {
    const { rerender } = render(
      <WeightsError result={gatedResult('not_granted', 'javascript:alert(1)')} />,
    )
    expect(screen.queryByRole('link')).toBeNull()
    rerender(<WeightsError result={gatedResult('not_granted', 'http://hub.local/org/gated')} />)
    expect(screen.queryByRole('link')).toBeNull()
    rerender(<WeightsError result={gatedResult('not_granted', null)} />)
    expect(screen.queryByRole('link')).toBeNull()
    expect(screen.getByRole('alert')).toHaveTextContent('申請制の重みを読めません')
  })

  it('申請制でない誤りは文だけを出す', () => {
    render(
      <WeightsError result={{ ok: false, reports: [], error: 'RuntimeError: 壊れたファイル' }} />,
    )
    expect(screen.getByRole('alert')).toHaveTextContent('RuntimeError: 壊れたファイル')
    expect(screen.queryByRole('link')).toBeNull()
  })
})
