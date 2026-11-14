import { expect, test } from '@playwright/test'

// 保存と読み込みを含むので, 書いた順に実行
test.describe.configure({ mode: 'serial' })

test('例を開くと, 誤り無しで形とパラメータ数が出る', async ({ page }) => {
  await page.goto('/')
  await page.getByRole('button', { name: '開く' }).click()
  await page.getByRole('button', { name: /^cnn_classifier/ }).click()

  // 確認と形の推定の結果がノードと上の欄に出る
  // (誤り 0 件は確認の結果が届く前にも出るので, 例に固有の値を先に待つ)
  await expect(page.locator('[data-id="head"]')).toContainText('[4, 10]', { timeout: 20_000 })
  await expect(page.locator('header')).toContainText('パラメータ数 5.1K')
  await expect(page.locator('header')).toContainText('誤り 0 件')
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
  await page.getByRole('button', { name: /^e2e_model \d/ }).click()
  await expect(page.locator('[data-id="linear1"]')).toBeVisible()
  await expect(page.locator('.react-flow__edge')).toHaveCount(2)
})

test('線を選んで Delete で外すと, 誤りが出る', async ({ page }) => {
  await page.goto('/')
  await page.getByRole('button', { name: '開く' }).click()
  await page.getByRole('button', { name: /^e2e_model \d/ }).click()
  await expect(page.locator('.react-flow__edge')).toHaveCount(2)

  // 入力から全結合への線を外すと, 必須の口が未接続の誤り
  await page.locator('.react-flow__edge').first().locator('path').first().click({ force: true })
  await page.keyboard.press('Delete')
  await expect(page.locator('.react-flow__edge')).toHaveCount(1)
  await expect(page.locator('header')).toContainText('誤り 1 件', { timeout: 20_000 })
})

test('合わない値はその場で理由を出し, ダイアログを閉じると元の所に戻る', async ({ page }) => {
  await page.goto('/')

  // 整数の引数に文字, 範囲の外の数を入れると理由が出て, 直すと消える
  await page.getByPlaceholder('部品を探す').fill('nn.Linear')
  await page
    .getByRole('button', { name: /^nn\.Linear/ })
    .first()
    .click()
  const field = page.getByLabel('in_features')
  await field.fill('abc')
  await field.press('Enter')
  await expect(page.getByText('整数を入れてください', { exact: true })).toBeVisible()
  await expect(field).toHaveAttribute('aria-invalid', 'true')
  await field.fill('0')
  await field.press('Enter')
  await expect(page.getByText('1 以上にしてください', { exact: true })).toBeVisible()
  await field.fill('8')
  await field.press('Enter')
  await expect(field).toHaveAttribute('aria-invalid', 'false')

  // ダイアログを閉じるボタンで閉じても, 定義を開いて上の操作が作り直されても,
  // 開いたボタンに戻る
  const open = page.getByRole('button', { name: '開く' })
  await open.click()
  await page.getByRole('dialog').getByRole('button', { name: '閉じる' }).click()
  await expect(page.getByRole('dialog')).toHaveCount(0)
  await expect(open).toBeFocused()
  await page.getByRole('button', { name: 'YAML' }).click()
  await page.getByRole('dialog').getByRole('button', { name: '閉じる' }).click()
  await expect(page.getByRole('button', { name: 'YAML' })).toBeFocused()
  page.once('dialog', (dialog) => void dialog.accept())
  await open.click()
  await page.getByRole('button', { name: /^e2e_model \d/ }).click()
  await expect(page.getByRole('dialog')).toHaveCount(0)
  await expect(page.getByRole('button', { name: '開く' })).toBeFocused()
})

test('元に戻す・やり直すで, 部品の追加と引数の変更を戻せる', async ({ page }) => {
  await page.goto('/')
  const undo = page.getByRole('button', { name: '元に戻す' })
  const redo = page.getByRole('button', { name: 'やり直す' })
  await expect(undo).toBeDisabled()

  // 部品を置いて引数を変え, 引数の変更から順に戻す
  await page.getByPlaceholder('部品を探す').fill('nn.Linear')
  await page
    .getByRole('button', { name: /^nn\.Linear/ })
    .first()
    .click()
  await expect(page.locator('[data-id="linear1"]')).toBeVisible()
  await page.waitForTimeout(600)
  await page.getByLabel('in_features').fill('8')
  await page.getByLabel('in_features').press('Enter')
  await page.locator('.react-flow__pane').click()
  await page.waitForTimeout(600)
  await page.keyboard.press('ControlOrMeta+z')
  await page.locator('[data-id="linear1"]').click()
  await expect(page.getByLabel('in_features')).toHaveValue('')
  await page.locator('.react-flow__pane').click()
  await undo.click()
  await expect(page.locator('[data-id="linear1"]')).toHaveCount(0)

  // やり直すと部品が戻り, 文字の欄の中の Ctrl+Z は定義を戻さない
  await redo.click()
  await expect(page.locator('[data-id="linear1"]')).toBeVisible()

  // 欄を通るだけ (値を変えない) では履歴に残らず, やり直しが消えない
  await page.locator('[data-id="linear1"]').click()
  await page.getByLabel('in_features').focus()
  await page.getByLabel('in_features').blur()
  await expect(redo).toBeEnabled()
  await page.locator('.react-flow__pane').click()
  await page.keyboard.press('ControlOrMeta+Shift+z')
  await page.locator('[data-id="linear1"]').click()
  await expect(page.getByLabel('in_features')).toHaveValue('8')
  await page.getByLabel('in_features').press('ControlOrMeta+z')
  await expect(page.locator('[data-id="linear1"]')).toBeVisible()
  await expect(redo).toBeDisabled()
})

test('整列で, つながりの順に左から右へ並べ直し, 元に戻せる', async ({ page }) => {
  await page.goto('/')
  await page.getByRole('button', { name: '開く' }).click()
  await page.getByRole('button', { name: /^residual_mlp/ }).click()
  const box = async (id: string) => (await page.locator(`[data-id="${id}"]`).boundingBox())!
  // 左にあるか (この例は折り返さない長さ. 飛び越すつなぎのために上下には
  // ずれる)
  const before = async (a: string, b: string) => (await box(a)).x < (await box(b)).x

  // 並べ直すと, 入力, つなぎ先の順に右へ
  await page.getByRole('button', { name: '整列' }).click()
  await page.waitForTimeout(400)
  expect(await before('x', 'fc1')).toBe(true)
  expect(await before('fc1', 'act')).toBe(true)
  expect(await before('act', 'add')).toBe(true)
  expect(await before('add', 'head')).toBe(true)
  await page.screenshot({ path: 'test-results/arranged.png' })

  // 元に戻すで, 並べ直す前の位置
  await page.getByRole('button', { name: '元に戻す' }).click()
  await expect(page.getByRole('button', { name: '元に戻す' })).toBeDisabled()
})

test('ノードを引数ごと複製し, 写して貼れる', async ({ page }) => {
  await page.goto('/')
  await page.getByPlaceholder('部品を探す').fill('nn.Linear')
  await page
    .getByRole('button', { name: /^nn\.Linear/ })
    .first()
    .click()
  await page.getByLabel('in_features').fill('16')
  await page.getByLabel('in_features').press('Enter')

  // 複製のボタンで, 引数の同じ写しが選ばれる
  await page.getByRole('button', { name: '複製' }).click()
  await expect(page.locator('[data-id="linear2"]')).toBeVisible()
  await expect(page.getByLabel('名前', { exact: true })).toHaveValue('linear2')
  await expect(page.getByLabel('in_features')).toHaveValue('16')

  // Ctrl+C で写し, 何も選ばずに Ctrl+V で貼る. Ctrl+D でも複製
  await page.locator('[data-id="linear1"]').click()
  await page.keyboard.press('ControlOrMeta+c')
  await page.locator('.react-flow__pane').click()
  await page.keyboard.press('ControlOrMeta+v')
  await expect(page.locator('[data-id="linear3"]')).toBeVisible()
  await page.keyboard.press('ControlOrMeta+d')
  await expect(page.locator('[data-id="linear4"]')).toBeVisible()
  await expect(page.getByLabel('in_features')).toHaveValue('16')
})

test('出力を選んだ後も欄が残らず, 形の推定の失敗を誤りに数え, 上書きは確かめる', async ({
  page,
  request,
}) => {
  await page.goto('/')

  // 出力を選んでから入力を選ぶと, 名前の欄は 1 つ
  await page.getByRole('button', { name: '出力', exact: true }).click()
  await page.getByRole('button', { name: '入力', exact: true }).click()
  await expect(page.getByLabel('名前', { exact: true })).toHaveCount(1)
  await expect(page.getByLabel('名前', { exact: true })).toHaveValue('input1')

  // 形の合わない全結合の層は, 形の推定の失敗として誤りに数える
  await page.getByRole('button', { name: '開く' }).click()
  page.once('dialog', (dialog) => void dialog.accept())
  await page.getByRole('button', { name: /^cnn_classifier/ }).click()
  await page.locator('[data-id="head"]').click()
  await page.getByLabel('in_features').fill('99')
  await page.getByLabel('in_features').press('Enter')
  await expect(page.locator('header')).toContainText('誤り 1 件', { timeout: 20_000 })
  await page.locator('.react-flow__pane').click()
  await expect(page.getByRole('region', { name: '形の推定で失敗したもの' })).toContainText('head')

  // 保存した定義 (前のテストで保存した e2e_model) と同じ名前で保存する
  // 場合は確かめ, 断れば保存しない
  const before = await (await request.get('/api/models/e2e_model')).text()
  await page.getByLabel('保存する名前').fill('e2e_model')
  let asked = ''
  page.once('dialog', (dialog) => {
    asked = dialog.message()
    void dialog.dismiss()
  })
  await page.getByRole('button', { name: /^保存/ }).click()
  await expect.poll(() => asked).toContain('e2e_model')
  await page.waitForTimeout(500)
  expect(await (await request.get('/api/models/e2e_model')).text()).toBe(before)
  await expect(page.getByRole('button', { name: '保存 *' })).toBeVisible()
})

test('狭い画面では上の操作をアイコンにし, ミニマップを出さない', async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 800 })
  await page.goto('/')
  // 文字は読み上げの名前として残り, 上の操作は 1 行 (高さが 1 行分)
  for (const name of ['整列', '入力', '出力', 'YAML', 'Python']) {
    await expect(page.getByRole('button', { name, exact: true })).toBeVisible()
  }
  expect((await page.locator('header').boundingBox())!.height).toBeLessThan(60)
  // 新規の直後 (部品が無い) は, 誤りの数ではなく始め方
  await expect(page.locator('header')).toContainText('部品なし')
  await expect(page.locator('.react-flow__minimap')).toBeHidden()
})

test('保存した定義を開くの一覧から消せる', async ({ page }) => {
  await page.goto('/')
  await page.getByRole('button', { name: '開く' }).click()
  const dialog = page.getByRole('dialog')
  await expect(dialog.getByRole('button', { name: /^e2e_model \d/ })).toBeVisible()

  // 確かめを断ると残り, 受けると一覧から外れる
  page.once('dialog', (confirm) => void confirm.dismiss())
  await dialog.getByRole('button', { name: 'e2e_model を消す' }).click()
  await expect(dialog.getByRole('button', { name: /^e2e_model \d/ })).toBeVisible()
  page.once('dialog', (confirm) => void confirm.accept())
  await dialog.getByRole('button', { name: 'e2e_model を消す' }).click()
  await expect(dialog.getByRole('button', { name: /^e2e_model \d/ })).toHaveCount(0)
})

test('組み立ての途中 (出力が無い, つないでいないノードがある) でも形を出す', async ({ page }) => {
  await page.goto('/')

  // 出力を置く前でも, つないだノードには形が出る
  await page.getByPlaceholder('部品を探す').fill('nn.Linear')
  await page
    .getByRole('button', { name: /^nn\.Linear/ })
    .first()
    .click()
  await page.getByLabel('in_features').fill('16')
  await page.getByLabel('in_features').press('Enter')
  await page.getByLabel('out_features').fill('4')
  await page.getByLabel('out_features').press('Enter')
  await page.getByLabel(/^input/).selectOption('x')
  await expect(page.locator('[data-id="linear1"]')).toContainText('[4, 4]', { timeout: 20_000 })

  // 複製したつないでいないノードがあっても, 他のノードの形は消えない
  await page.getByRole('button', { name: '複製' }).click()
  await expect(page.locator('[data-id="linear2"]')).toBeVisible()
  await page.waitForTimeout(1500)
  await expect(page.locator('[data-id="linear1"]')).toContainText('[4, 4]')
  await expect(page.locator('header')).toContainText('パラメータ数 68')
})

test('複数つなぐ口のつなぎ先を並べ替えられる', async ({ page }) => {
  await page.goto('/')
  await page.getByPlaceholder('部品を探す').fill('op.Concat')
  await page
    .getByRole('button', { name: /^op\.Concat/ })
    .first()
    .click()
  await page.getByPlaceholder('部品を探す').fill('nn.ReLU')
  await page
    .getByRole('button', { name: /^nn\.ReLU / })
    .first()
    .click()
  await page.getByLabel(/^input/).selectOption('x')

  // concat1 に x と relu1 の順でつなぎ, relu1 を上へ
  await page.locator('[data-id="concat1"]').click()
  await page.getByLabel('xs につなぎ先を加える').selectOption('x')
  await page.getByLabel('xs につなぎ先を加える').selectOption('relu1')
  await page.getByRole('button', { name: 'xs の 2 番目のつなぎ先 relu1 を上へ' }).click()
  // 動かした先の行のボタンにフォーカスが移る (先頭なので下へ)
  await expect(
    page.getByRole('button', { name: 'xs の 1 番目のつなぎ先 relu1 を下へ' }),
  ).toBeFocused()
  await expect(
    page.getByRole('button', { name: 'xs の 1 番目のつなぎ先 relu1 を外す' }),
  ).toBeVisible()
  await expect(page.getByRole('button', { name: 'xs の 2 番目のつなぎ先 x を外す' })).toBeVisible()
})

test('大きすぎて形を推定しない場合は, 上の欄で知らせる', async ({ page }) => {
  await page.goto('/')
  await page.locator('[data-id="x"]').click()
  await page.getByLabel('shape').fill('[100000, 100000]')
  await page.getByLabel('shape').press('Enter')
  await expect(page.locator('header')).toContainText('形は未推定', { timeout: 20_000 })
  await page.getByLabel('shape').fill('[4, 16]')
  await page.getByLabel('shape').press('Enter')
  await expect(page.locator('header')).not.toContainText('形は未推定', { timeout: 20_000 })
})

test('キーボードだけで部品を置き, 名前を変えて戻し, Ctrl+S で保存する', async ({ page }) => {
  await page.goto('/')

  // 検索欄の Enter で一番上の部品を置き, 詳細の最初の欄に移る
  await page.getByPlaceholder('部品を探す').fill('nn.Linear')
  await page.getByPlaceholder('部品を探す').press('Enter')
  await expect(page.locator('[data-id="linear1"]')).toBeVisible()
  await expect(page.getByLabel('名前', { exact: true })).toBeFocused()

  // 名前を Enter で変えても名前の欄に残り, 元に戻すと元の名前を選んだまま
  await page.getByLabel('名前', { exact: true }).fill('fc')
  await page.getByLabel('名前', { exact: true }).press('Enter')
  await expect(page.getByLabel('名前', { exact: true })).toBeFocused()
  await expect(page.getByLabel('名前', { exact: true })).toHaveValue('fc')
  await page.locator('.react-flow__pane').click()
  await page.locator('[data-id="fc"]').click()
  await page.keyboard.press('ControlOrMeta+z')
  await expect(page.getByLabel('名前', { exact: true })).toHaveValue('linear1')

  // 保存する名前の欄の中でも Ctrl+S で保存する (保存していない変更の印が消える)
  await page.getByLabel('保存する名前').fill('keyboard_model')
  await page.getByLabel('保存する名前').press('ControlOrMeta+s')
  await expect(page.getByRole('button', { name: '保存', exact: true })).toBeVisible()
})
