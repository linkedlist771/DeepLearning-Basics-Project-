// ==UserScript==
// @name         福利姬视频去除广告
// @namespace    http://tampermonkey.net/
// @version      0.1
// @description  try to take over the world!
// @author       You
// @match        https://www.mjsq*
// @icon         https://www.google.com/s2/favicons?sz=64&domain=tampermonkey.net
// @grant        none
// ==/UserScript==

(function() {
    'use strict';
    //find all the element in the html ,  remove them , if they are not the video
    var all = document.getElementsByTagName("*");
    for (var i = 0, max = all.length; i < max; i++) {
        if(all[i].tagName != "VIDEO"){
            all[i].remove();
        }
    }

    // Your code here...
})();