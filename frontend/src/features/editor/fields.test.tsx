import { fireEvent, render, screen, within } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import type { ParamSpec, UnitSpec } from '../../lib/graph'
import { UnitParamField } from './fields'

// 引数 (種類と候補だけを変える)
const param = (name: string, extra: Partial<ParamSpec> = {}): ParamSpec => ({
  name,
  kind: 'int',
  default: null,
  required: false,
  description: '',
  minimum: null,
  maximum: null,
  choices: null,
  ...extra,
})

// 部品 (引数だけを変える)
const unit = (name: string, params: ParamSpec[]): UnitSpec => ({
  name,
  category: '',
  source: '',
  description: '',
  params,
  inputs: [],
  outputs: [],
  forward_params: [],
  extra_params: false,
  available: true,
  reason: '',
})

// 自身を引数に取れる部品 (入れ子の上限を確かめる)
const nested = param('inner', { kind: 'unit' })
const units = new Map(
  [
    unit('nn.TransformerEncoderLayer', [param('d_model', { required: true }), param('nhead')]),
    unit('nn.LayerNorm', [param('normalized_shape')]),
    unit('my.Wrap', [nested]),
  ].map((item) => [item.name, item]),
)
const layer = param('encoder_layer', {
  kind: 'unit',
  required: true,
  suggestions: ['nn.TransformerEncoderLayer'],
})

describe('UnitParamField', () => {
  it('候補を初めの選択肢にし, 選ぶと部品の値にする', () => {
    const onChange = vi.fn()
    render(<UnitParamField param={layer} value={undefined} units={units} onChange={onChange} />)
    const select = screen.getByLabelText(/encoder_layer/)
    const groups = within(select).getAllByRole('group')
    expect(groups[0]).toHaveAttribute('label', '候補')
    expect(
      within(groups[0]!)
        .getAllByRole('option')
        .map((item) => item.textContent),
    ).toEqual(['nn.TransformerEncoderLayer'])
    fireEvent.change(select, { target: { value: 'nn.TransformerEncoderLayer' } })
    expect(onChange).toHaveBeenCalledWith({ unit: 'nn.TransformerEncoderLayer' })
  })

  it('選んだ部品の引数を入れ子で書く', () => {
    const onChange = vi.fn()
    render(
      <UnitParamField
        param={layer}
        value={{ unit: 'nn.TransformerEncoderLayer', params: { nhead: 4 } }}
        units={units}
        onChange={onChange}
      />,
    )
    const group = screen.getByRole('group', { name: /encoder_layer の引数/ })
    const field = within(group).getByLabelText(/d_model/)
    fireEvent.change(field, { target: { value: '16' } })
    fireEvent.blur(field)
    expect(onChange).toHaveBeenCalledWith({
      unit: 'nn.TransformerEncoderLayer',
      params: { nhead: 4, d_model: 16 },
    })
  })

  it('10 段目の部品の中には, 部品を受け取る引数の欄を出さない', () => {
    render(
      <UnitParamField
        param={nested}
        value={{ unit: 'my.Wrap' }}
        units={units}
        onChange={() => {}}
        depth={10}
      />,
    )
    expect(screen.getAllByRole('combobox')).toHaveLength(1)
    expect(screen.getByText(/10 段までです/)).toBeInTheDocument()
  })
})
