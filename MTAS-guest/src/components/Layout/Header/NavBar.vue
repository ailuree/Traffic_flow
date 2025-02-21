<template>
  <div class="menu">
    <div class="menu-item title-index">
      {{ username }}
    </div>
    <template v-for="menu of menuList" :key="menu.name">
      <div v-if="menu" class="menu-item" :class="{ active: '首页' === menu.name }">
        <div :to="menu.path" class="menu-btn" @click='go(menu.path)'>
          <svg-icon :icon-class="menu.icon"></svg-icon> {{ menu.name }}
        </div>
      </div>

    </template>
  </div>
</template>

<script setup lang="ts">
import router from "@/router";
import useStore from "@/store";

// 获取昵称
const { user } = useStore();
const username: String = user.getUser();


// 列表
const menuList = [
  {
    name: "首页",
    icon: "home",
    path: "/#"
  },
];

</script>

<style lang="scss" scoped>
.user-avatar {
  display: inline-block;
  position: relative;
  top: 0.3rem;
  width: 24px;
  height: 24px;
  border-radius: 50%;
  cursor: pointer;
}

.menu {
  display: flex;
  align-items: center;
  height: 100%;
}

.menu-item {
  position: relative;
  display: inline-block;
  padding: 0 0.625rem;
  letter-spacing: 0.0625rem;
  font-size: 17px;
  text-align: center;

  &:not(.title-index) .menu-btn::before {
    content: "";
    position: absolute;
    width: 0;
    height: 0.1875rem;
    bottom: 0;
    border-radius: 0.125rem;
    left: 50%;
    transform: translateX(-50%);
    background-color: currentColor;
    transition: all 0.4s ease-in-out 0s;
  }

  &:hover .submenu {
    display: block;
  }
}

.menu-item.active:not(.dropdown) .menu-btn::before,
.menu-item:not(.dropdown):hover .menu-btn::before {
  width: 70%;
}

.submenu {
  display: none;
  position: absolute;
  left: 7px;
  width: max-content;
  margin-top: 0.5rem;
  padding: 0;
  background: var(--grey-9-a5);
  box-shadow: 0 0.3125rem 1.25rem -0.25rem var(--grey-9-a1);
  border-radius: 0.625rem 0;
  animation: slideUpIn 0.3s;

  &::before {
    position: absolute;
    top: -1.25rem;
    left: 0;
    width: 100%;
    height: 2.5rem;
    content: "";
  }
}

.subitem {
  display: block;
  font-size: 1rem;

  &:first-child {
    border-radius: 0.625rem 0 0 0;
  }

  &:last-child {
    border-radius: 0 0 0.625rem 0;
  }

  .link {
    display: inline-block;
    padding: 0.3rem 0.7rem;
    width: 100%;
    text-shadow: none;
  }

  &:hover .link {
    transform: translateX(0.3rem);
  }
}

.submenu .subitem.active,
.submenu .subitem:hover {
  color: var(--grey-0);
  background-image: linear-gradient(to right, var(--color-pink) 0, var(--color-orange) 100%);
  box-shadow: 0 0 0.75rem var(--color-pink-a3);
}

.sub.menu .submenu {
  background-color: var(--grey-1);
}

.drop::after {
  content: "";
  display: inline-block;
  vertical-align: middle;
  border: 0.3rem solid transparent;
  border-top-color: currentColor;
  border-bottom: 0;
}

@media (max-width: 865px) {
  .menu {
    justify-content: center;
  }

  .menu .menu-item {
    display: none;
  }

  .menu .title-index {
    display: block;
  }
}
</style>
