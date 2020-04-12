var jq = document.createElement('script');
jq.src = "https://code.jquery.com/jquery-3.3.1.min.js";  /* Include any online jquery library you need */
document.getElementsByTagName('head')[0].appendChild(jq);
function sleep(ms) {
  return new Promise(resolve =>
      setTimeout(resolve, ms)
  )
}
sleep(500).then(()=>{
   $.noConflict
})

var str =
"        --dataroot          /opt/data/private/datasets/vimeo_septuplet\n" +
    "        --name              vimeo_rbpn\n" +
    "        --model             rbpn\n" +
    "        --display_freq      2400\n" +
    "        --print_freq        2400\n" +
    "        --save_epoch_freq   5\n" +
    "        --gpu_ids           0,1\n" +
    "        --batch_size        8\n" +
    "        --suffix            small_64_64_16_0405_2203\n" +
    "        --cl                64\n" +
    "        --cm                64\n" +
    "        --ch                16";


str = str.split("\n");
for(var i=0; i<str.length; i++){
    str[i] = str[i].trim().split(/\s+/);
    str[i][0] = str[i][0].slice(2);
    if (i>0){
        $("i.el-icon-plus").click();
    }
}

sleep(1000).then(()=>{
    var alltxt = $("input[placeholder=key]");
    $.each(alltxt, function(i, value){
       $(this).val(str[i][0]);
    });
    alltxt = $("input[placeholder=value]");
    $.each(alltxt, function(i, value){
       $(this).val(str[i][1]);
    });
})





