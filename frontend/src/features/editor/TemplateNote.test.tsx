import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import { TemplateNote } from './ProjectDialogs'

describe('TemplateNote', () => {
  it('開いた雛形は知らせだけを出す', () => {
    render(
      <TemplateNote
        template={{ example: 'two_tower_retrieval', problem: '推薦', opened: true, reason: null }}
        opening={false}
        error={null}
        onOpen={() => {}}
      />,
    )
    expect(screen.getByRole('status')).toHaveTextContent(
      '問題 推薦 の受け持ちに合う雛形 two_tower_retrieval を開きました',
    )
    expect(screen.queryByRole('button')).toBeNull()
  })

  it('開いていない雛形は理由と「雛形を開く」を出す', () => {
    const onOpen = vi.fn()
    render(
      <TemplateNote
        template={{
          example: 'unet_segmentation',
          problem: null,
          opened: false,
          reason: '今の定義があるので開いていません',
        }}
        opening={false}
        error={null}
        onOpen={onOpen}
      />,
    )
    expect(screen.getByRole('status')).toHaveTextContent(
      '問題の受け持ちに合う雛形 unet_segmentation があります (今の定義があるので開いていません)',
    )
    fireEvent.click(screen.getByRole('button', { name: '雛形 unet_segmentation を開く' }))
    expect(onOpen).toHaveBeenCalledTimes(1)
  })

  it('読んでいる間は押し直せない', () => {
    render(
      <TemplateNote
        template={{ example: 'vae', problem: null, opened: false, reason: null }}
        opening
        error={null}
        onOpen={() => {}}
      />,
    )
    expect(screen.getByRole('button', { name: '開いています…' })).toBeDisabled()
  })
})
