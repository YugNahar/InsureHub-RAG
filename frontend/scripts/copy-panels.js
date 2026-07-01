import 'dotenv/config';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, '..');
const apiUrl = (process.env.VITE_API_BASE_URL || '').replace(/\/$/, '');
const panelFiles = ['auth.html', 'admin.html', 'agent-dashboard.html'];

let copied = 0;
for (const file of panelFiles) {
  const src = path.join(root, 'panels', file);
  if (!fs.existsSync(src)) {
    console.error(`[copy-panels] ERROR: panels/${file} not found`);
    process.exit(1);
  }
  const content = fs.readFileSync(src, 'utf8').replace(/__API_URL__/g, apiUrl);
  fs.writeFileSync(path.join(root, 'dist', file), content);
  console.log(`[copy-panels] ✓ ${file}`);
  copied++;
}
console.log(`[copy-panels] Done — ${copied} panels copied, API: ${apiUrl || '(none — users enter it at login)'}`);
