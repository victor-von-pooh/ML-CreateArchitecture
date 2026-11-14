import { test, type Page } from '@playwright/test'
test.describe.configure({ mode: 'serial' })
const shot = (p: Page, n: string) => p.screenshot({ path: `test-results/x2-${n}.png` })
async function add(page: Page, name: string) {
  await page.getByPlaceholder('部品を探す').fill(name)
  await page.getByRole('button', { name: new RegExp('^' + name.replace('.', '\\.')) }).first().click()
  await page.waitForTimeout(300)
}
test('gnn', async ({ page }) => {
  await page.goto('/')
  await page.getByRole('button', { name: '入力', exact: true }).click()
  await shot(page, 'input-panel')
  console.log(await page.locator('aside').last().innerText())
  await add(page, 'pyg.GCNConv')
  await shot(page, 'gcn-panel')
  console.log('---GCN---\n' + (await page.locator('aside').last().innerText()))
  const html = await page.locator('aside').last().innerHTML()
  console.log(html.slice(0, 3000))
})
