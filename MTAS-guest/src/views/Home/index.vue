<template>
  <Provider>
    <div class="app-wrapper">
      <main class="main-wrapper">
        <Header></Header>

        <!--背景轮播-->
        <Images></Images>
        <!-- 品牌标题 -->
        <Brand></Brand>
        <div class="bg">
          <Mycontent></Mycontent>
        </div>
        <Footer></Footer>
      </main>
    </div>
  </Provider>
</template>


<script setup lang="ts">
import Brand from "./Brand/index.vue";
import Images from "./Swiper/Images.vue";
import Mycontent from "./content/content.vue"
</script>

<style lang="scss" scoped>
.mt {
  margin-top: 1rem;
  padding-bottom: 1.75rem;
}

.app-wrapper {
  position: relative;
  min-height: 100vh;
  min-width: 320px;
}

.main-wrapper {
  display: flex;
  flex-direction: column;
  width: 100%;
  padding: 0 0 8rem;
}
</style>
