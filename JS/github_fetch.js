async function fetchAndRenderTree(url, parentUl) {
  const response = await fetch(url);
  const data = await response.json();

  for (const item of data) {
    const li = document.createElement('li');
    li.className = item.type === 'dir' ? 'folder' : 'file';

    if (item.type === 'dir') {
      li.textContent = item.name;
      const subUl = document.createElement('ul');
      await fetchAndRenderTree(item.url, subUl); // Recursive fetch
      li.appendChild(subUl);
    } else {
      const a = document.createElement('a');
      a.href = item.html_url;
      a.textContent = item.name;
      a.target = '_blank';
      li.appendChild(a);
    }
    parentUl.appendChild(li);
  }
}

(async () => {
  const toc = document.getElementById('FileTreeContents');
  toc.innerHTML = '';
  const ul = document.createElement('ul');
  ul.className = 'file-tree';
  await fetchAndRenderTree('https://api.github.com/repos/devCacti/snake-rl/contents/', ul);
  toc.appendChild(ul);
})();