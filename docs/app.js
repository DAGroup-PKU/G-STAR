'use strict';
const models=[
['GPT Image 1','—','—','0.84','85.15','8.07','0.956','54.07','other'],
['Seedream 3.0','—','—','0.84','88.27','7.62','0.896','—','other'],
['HiDream-I1-Full','16×','17B','0.83','85.89','7.26','0.543','43.08','other'],
['FLUX.1 [dev]','16×','12B','0.66','83.84','6.85','0.607','21.58','other'],
['BAGEL','16×','14B','0.88','85.07','6.20','0.373','42.04','other'],
['Z-Image','16×','6B','0.84','88.14','7.56','0.935','49.10','other'],
['Qwen-Image','16×','20B','0.87','88.32','7.41','0.943','49.23','other'],
['OmniGen2','16×','4B','0.86','83.57','6.48','0.561','40.70','other'],
['SANA-1.5','32×','4.8B','0.81','84.70','6.51','0.050','35.14','other'],
['Janus-Pro-7B','16×','7B','0.80','84.19','6.21','0.019','23.53','ar'],
['GLM-Image','16×','16B','0.72','84.78','7.32','0.952','48.19','ar'],
['Emu3.5','16×','34.1B','0.86','88.26','7.52','0.976','49.89','ar'],
['BitDance','16×','14B','0.86','88.28','7.19','0.837','45.19','ar'],
['G-STAR-Pro-8B','32×','8B','0.84','83.38','6.98','0.935','50.07','ours'],
['G-STAR-Flash-1.7B','64×*','1.7B','0.76','80.18','6.81','0.853','46.88','ours']];
function renderTable(filter){const tbody=document.querySelector('#benchmark tbody');tbody.replaceChildren();for(const row of models){if(filter==='ours'&&row[8]!=='ours')continue;if(filter==='ar'&&row[8]==='other')continue;const tr=document.createElement('tr');if(row[8]==='ours')tr.className='ours';row.slice(0,8).forEach((text,i)=>{const cell=document.createElement(i===0?'th':'td');if(i===0)cell.scope='row';cell.textContent=text;tr.append(cell)});tbody.append(tr)}}renderTable('all');
for(const button of document.querySelectorAll('[data-filter]'))button.addEventListener('click',()=>{for(const other of document.querySelectorAll('[data-filter]')){other.classList.toggle('active',other===button);other.setAttribute('aria-pressed',String(other===button))}renderTable(button.dataset.filter)});
document.querySelector('#scale').addEventListener('change',event=>{const packed=event.target.value==='64';document.querySelector('#steps').textContent=packed?'256':'1,024';document.querySelector('#groups').textContent=packed?'32':'8'});
for(const input of document.querySelectorAll('.edits input'))input.addEventListener('input',()=>input.closest('article').querySelector('.comparison').style.setProperty('--split',input.value+'%'));
const lightbox=document.querySelector('#lightbox');for(const button of document.querySelectorAll('[data-image]'))button.addEventListener('click',()=>{lightbox.querySelector('img').src=button.dataset.image;lightbox.querySelector('img').alt=button.querySelector('img').alt;lightbox.querySelector('p').textContent=button.dataset.caption;lightbox.showModal()});document.querySelector('#close-lightbox').addEventListener('click',()=>lightbox.close());lightbox.addEventListener('click',event=>{if(event.target===lightbox){const rect=lightbox.getBoundingClientRect();if(event.clientX<rect.left||event.clientX>rect.right||event.clientY<rect.top||event.clientY>rect.bottom)lightbox.close()}});
