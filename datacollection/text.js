const got = require('got');
const jsdom = require("jsdom");
const { JSDOM } = jsdom;

const urls = [
  'https://de.wikipedia.org/wiki/Brot',
  'https://de.wikipedia.org/wiki/Traktor',
  'https://de.wikipedia.org/wiki/H%C3%A4keln',
  'https://de.wikipedia.org/wiki/Polizei',
  'https://de.wikipedia.org/wiki/Philosophie',
]

async function getSentences(url) {
  const response = await got(url);
  const dom = new JSDOM(response.body);

  // Create an Array out of the HTML Elements for filtering using spread syntax.
  const nodeList = [...dom.window.document.getElementById('mw-content-text').getElementsByTagName('p')];
  return nodeList.map(node => node.textContent)  
}

(async () => {
  const dataEntries = []
  for(let i=0;i<urls.length;i++) {
    const data = await getSentences(urls[i])
    dataEntries.push(data)
  }
  console.log(dataEntries)

  const createCsvWriter = require('csv-writer').createObjectCsvWriter;
  const csvWriter = createCsvWriter({
      path: '../data/wikipedia.csv',
      header: [
          {id: 'text', title: 'text'},
          {id: 'label', title: 'label'}
      ]
  });

  const records = []
  dataEntries.forEach((data, idx) => {
    data.forEach(sentence => {
      records.push({
        text: sentence.trim(),
        label: idx,
      })
    })
  })


  csvWriter.writeRecords(records)
})()