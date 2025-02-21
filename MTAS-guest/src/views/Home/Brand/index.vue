<template>
  <div class="brand-container">
    <div class="brand">
      <!-- 标题 -->
      <p class="artboard">车流量检测系统</p>

      <!-- 打字机 -->
      <div style="display: flex; align-items: center;">
        <span ref="myText" class="text title-print">
        </span><span class="easy-typed-cursor">|</span>
      </div>



    </div>
    <!-- 波浪 -->
    <Waves></Waves>
    <!-- 向下按钮 -->
    <svg-icon class="arrow-down" icon-class="down" size="32px" @click="scrollDown"></svg-icon>
  </div>
</template>

<script setup lang="ts">

import { ref, onMounted } from 'vue';

const txt = [
  " 预测未来的最好方法就是创造未来",
  " 挫折会来，也会过去；热泪会留下，也会收起！",
  " 以不息为体，以日新为道 ",
  " 我们活过的刹那，前后皆是阳光",
  " 无冥冥之志者，无昭昭之明；无惛惛之事者，无赫赫之功",
];

const myText = ref<HTMLSpanElement | null>(null);
let index = 0;
let xiaBiao = 0;
let huan = true;

onMounted(() => {

  setInterval(() => {
    const myTextElement = myText.value;
    if (!myTextElement) {
      console.error("Target element is null.");
      return;
    }
    if (huan) {
      myTextElement.textContent = txt[xiaBiao].slice(1, ++index);
    } else {
      myTextElement.textContent = txt[xiaBiao].slice(1, index--);
    }

    if (index === txt[xiaBiao].length + 3) {
      huan = false;
    } else if (index < 0) {
      index = 0;
      huan = true;
      xiaBiao++;
      if (xiaBiao >= txt.length) {
        xiaBiao = 0;
      }
    }
  }, 200);
});



const scrollDown = () => {
  window.scrollTo({
    behavior: "smooth",
    top: document.documentElement.clientHeight,
  });
};



</script>

<style lang="scss" scoped>
@import "@/assets/styles/mixin.scss";

body {
  cursor: url("../icons/normal.cur"), default;
  background: var(--grey-0);
  color: var(--text-color);
  font-family: Mulish, -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif;
  font-size: 1em;
  overflow-x: hidden;
  line-height: 2;
}

.brand-container {
  @include flex;
  flex-direction: column;
  position: relative;
  width: 100%;
  height: 100vh;
  min-height: 10rem;
  color: var(--header-text-color);
}

.brand {
  @include flex;
  flex-direction: column;
  position: fixed;
  z-index: -1;

  .artboard {
    font-family: "Fredericka the Great", Mulish, -apple-system, "PingFang SC", "Microsoft YaHei",
      sans-serif;
    //font-family: Mulish, -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif;
    font-size: 3.5em;
    line-height: 1.2;
    animation: titleScale 1s;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.9);

  }

  .title-print {
    letter-spacing: 0.1em;
    font-family: Mulish, -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif;
    //font-size: 1em;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.9);
    color: rgb(255, 255, 255);
  }
}

.easy-typed-cursor {
  margin-left: 0.625rem;
  opacity: 1;
  -webkit-animation: blink 0.7s infinite;
  -moz-animation: blink 0.7s infinite;
  animation: blink 0.7s infinite;
}

.arrow-down {
  position: absolute;
  bottom: 70px;
  -webkit-animation: arrow-shake 1.5s ease-out infinite;
  animation: arrow-shake 1.5s ease-out infinite;
  cursor: pointer;
  z-index: 8;
}

@media (max-width: 767px) {
  .brand-container {
    padding: 3rem 0.5rem 0;
  }
}

@media (min-width: 760px) {
  .title-print {
    font-size: 1.5rem;
  }
}

@keyframes arrow-shake {
  0% {
    opacity: 1;
    transform: translateY(0);
  }

  30% {
    opacity: 0.5;
    transform: translateY(25px);
  }

  100% {
    opacity: 1;
    transform: translateY(0);
  }
}

@keyframes blink {
  0% {
    opacity: 0;
  }

  100% {
    opacity: 1;
  }
}
</style>
