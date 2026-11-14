import { expect, test } from '@playwright/test'

// 保存と読み込みを含むので, 書いた順に実行
test.describe.configure({ mode: 'serial' })

test('例を開くと, 誤り無しで形とパラメータ数が出る', async ({ page }) => {
  await page.goto('/')
  await page.getByRole('button', { name: '開く' }).click()
  await page.getByRole('button', { name: /^cnn_classifier/ }).click()

  // 確認と形の推定の結果がノードと上の欄に出る
  await expect(page.locator('header')).toContainText('誤り 0 件', { timeout: 20_000 })
  await expect(page.locator('header')).toContainText('パラメータ数 5.1K')
  await expect(page.locator('[data-id="head"]')).toContainText('[4, 10]')
})

test('部品を置いて選択の欄でつなぎ, 保存して開き直す', async ({ page }) => {
  await page.goto('/')

  // 全結合の層を置いて引数を入れる
  await page.getByPlaceholder('部品を探す').fill('nn.Linear')
  await page
    .getByRole('button', { name: /^nn\.Linear/ })
    .first()
    .click()
  await page.getByLabel('in_features').fill('16')
  await page.getByLabel('in_features').press('Enter')
  await page.getByLabel('out_features').fill('4')
  await page.getByLabel('out_features').press('Enter')

  // 入力の口と出力を, 線を引かずに選択の欄でつなぐ
  await page.getByLabel(/^input/).selectOption('x')
  await page.getByRole('button', { name: '出力' }).click()
  await page.getByLabel(/つなぎ先/).selectOption('linear1')
  await expect(page.locator('.react-flow__edge')).toHaveCount(2)
  await expect(page.locator('header')).toContainText('パラメータ数 68', { timeout: 20_000 })
  await expect(page.locator('header')).toContainText('誤り 0 件')

  // 名前を付けて保存すると, 保存していない変更の印が消える
  await page.getByLabel('保存する名前').fill('e2e_model')
  await page.getByRole('button', { name: /^保存/ }).click()
  await expect(page.getByRole('button', { name: '保存', exact: true })).toBeVisible()

  // 新しく作ってから開き直すと, 同じ定義
  await page.getByRole('button', { name: '新規' }).click()
  await expect(page.locator('[data-id="linear1"]')).toHaveCount(0)
  await page.getByRole('button', { name: '開く' }).click()
  await page.getByRole('button', { name: /^e2e_model/ }).click()
  await expect(page.locator('[data-id="linear1"]')).toBeVisible()
  await expect(page.locator('.react-flow__edge')).toHaveCount(2)
})

test('線を選んで Delete で外すと, 誤りが出る', async ({ page }) => {
  await page.goto('/')
  await page.getByRole('button', { name: '開く' }).click()
  await page.getByRole('button', { name: /^e2e_model/ }).click()
  await expect(page.locator('.react-flow__edge')).toHaveCount(2)

  // 入力から全結合への線を外すと, 必須の口が未接続の誤り
  await page.locator('.react-flow__edge').first().locator('path').first().click({ force: true })
  await page.keyboard.press('Delete')
  await expect(page.locator('.react-flow__edge')).toHaveCount(1)
  await expect(page.locator('header')).toContainText('誤り 1 件', { timeout: 20_000 })
})
