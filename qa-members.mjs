export default async function run(page) {
  await page.getByLabel('email').fill('test.admin@example.com');
  await page.getByLabel('password').fill('Test12345!');
  await page.getByRole('button', { name: 'Log In' }).click();
  await page.waitForURL(/\/projects\//, { timeout: 15000 });
  const afterLogin = { url: page.url(), text: await page.locator('body').innerText() };
  await page.goto('http://localhost:8080/projects/1/settings/members');
  await page.waitForTimeout(1000);
  return {
    afterLogin: { url: afterLogin.url, hasProjectsText: afterLogin.text.includes('Projects') },
    members: {
      url: page.url(),
      hasMembersText: (await page.locator('body').innerText()).includes('Members'),
      bodyText: (await page.locator('body').innerText()).slice(0, 1200),
    },
  };
}
