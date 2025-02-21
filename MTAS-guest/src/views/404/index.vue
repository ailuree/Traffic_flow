<template>
  <div class="page-header" style="min-width: 400px;">
    <h1 class="page-title">页面不存在</h1>
    <img class="page-cover" src="../../assets/images/404.gif"
      alt="">
    <Waves></Waves>
  </div>
  <div class="notfound"><text>Σ( ° △ °|||)︴404-网页为找到！</text></div>
  <div class="notfound">来自：pan的抱歉！</div>
  <div class="notfound"><router-link to="/">返回主页</router-link></div>
  
  <!-- <div class="bg">
    <div class="page-container">
      <div class="notfound">Σ( ° △ °|||)︴404-网页为找到！</div>
      <div class="notfound">来自：pan的抱歉！</div>
    </div>
  </div> -->
</template>

<script setup lang="ts">
</script>

<style scoped>
.notfound {
  width: 20.75rem;
  height: 0.1rem;
  /* background: url(../../assets/images/404.gif) no-repeat bottom; */
  text-align: center;
  margin: 3rem auto;
}



.apology{
  width: 18.75rem;
  height: 0.1rem;
  /* background: url(../../assets/images/404.gif) no-repeat bottom; */
  text-align: center;
  margin: 6.25rem auto;
}

 



</style>