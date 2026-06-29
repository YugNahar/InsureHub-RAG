const fs = require('fs');
const path = require('path');

// Same env var the main Layla frontend uses
const API_URL = (process.env.VITE_API_BASE_URL || '').replace(/\/$/, '');

fs.mkdirSync('dist', { recursive: true });

const files = ['auth.html', 'admin.html', 'agent-dashboard.html'];
for (const file of files) {
  const content = fs.readFileSync(file, 'utf8').replace(/__API_URL__/g, API_URL);
  fs.writeFileSync(path.join('dist', file), content);
}

console.log(`Built ${files.length} panels — API_URL: ${API_URL || '(none — users will enter it on login)'}`);
