import { expect, test } from '@playwright/test'

// 起動の設定 (e2e/serve.sh の ML_CREATEARCHITECTURE_MODULES) に書いた見本のモジュールの
// 自作の部品を置き, 登録されていない部品には起動の設定の案内を出す
test.describe.configure({ mode: 'serial' })

test('起動の設定のモジュールの自作の部品が, モジュールのまとまりに出て置ける', async ({ page }) => {
  await page.goto('/')

  // モジュールの名前を付けたまとまりが開いた状態で出る
  const group = page.getByRole('button', { name: /^自作 \(e2e_lab_units\)/ })
  await expect(group).toHaveAttribute('aria-expanded', 'true')

  // 置いて入力と出力につなぐと, 誤り無しで形が出る
  await page.getByRole('button', { name: /^e2e_lab\.Shift/ }).click()
  await page.getByLabel(/^x/).selectOption('x')
  await page.getByRole('button', { name: '出力', exact: true }).click()
  await page.getByLabel(/つなぎ先/).selectOption('shift1')
  await expect(page.locator('header')).toContainText('誤り 0 件', { timeout: 20_000 })
  await expect(page.locator('[data-id="shift1"]')).toContainText('[')
})

test('登録されていない部品の定義を開くと, 起動の設定の案内が出る', async ({ page, request }) => {
  // 読み込んでいないモジュールの名前を部品に書いた定義を保存
  const spec = {
    format_version: 1,
    inputs: { x: { shape: [2, 3] } },
    nodes: { block: { unit: 'not_listed_lab.Block', inputs: 'x' } },
    outputs: { y: 'block' },
  }
  const saved = await request.put('/api/models/e2e_unregistered', { data: { spec } })
  expect(saved.ok()).toBe(true)

  // 開いてノードを選ぶと, 誤りと起動の設定の案内 (環境変数と読み込んでいるモジュール)
  await page.goto('/')
  await page.getByRole('button', { name: '開く' }).click()
  await page.getByRole('button', { name: /^e2e_unregistered \d/ }).click()
  await page.locator('[data-id="block"]').click()
  await expect(
    page.getByText(/この部品 \('not_listed_lab\.Block'\) は登録されていません/),
  ).toBeVisible({
    timeout: 20_000,
  })
  const hint = page.getByRole('note', { name: '自作の部品の足し方' })
  await expect(hint).toContainText('ML_CREATEARCHITECTURE_MODULES')
  await expect(hint).toContainText('今読み込んでいるモジュール: e2e_lab_units')
})
