import "@/assets/fonts/font.css"; // 引入字体样式
import "@/assets/styles/index.scss"; // 引入全局样式
import SvgIcon from "@/components/SvgIcon/index.vue"; // 引入SvgIcon组件
import * as directive from "@/directive"; // 引入自定义指令

import router from "@/router"; // 引入路由配置
import { titleChange } from "@/utils/title"; // 引入标题变更函数
import createKatexPlugin from "@kangc/v-md-editor/lib/plugins/katex/cdn"; // 引入katex插件
import createTodoListPlugin from "@kangc/v-md-editor/lib/plugins/todo-list/index"; // 引入todo-list插件
import "@kangc/v-md-editor/lib/plugins/todo-list/todo-list.css"; // 引入todo-list样式
import VMdPreview from "@kangc/v-md-editor/lib/preview"; // 引入v-md-editor预览模块
import "@kangc/v-md-editor/lib/theme/style/vuepress.css"; // 引入v-md-editor主题样式
import vuepressTheme from "@kangc/v-md-editor/lib/theme/vuepress.js"; // 引入v-md-editor vuepress主题
import naive from "naive-ui"; // 引入Naive UI组件库
import "nprogress/nprogress.css"; // 引入NProgress加载条样式
import { createPinia } from "pinia"; // 引入Pinia状态管理库
import piniaPluginPersistedstate from "pinia-plugin-persistedstate"; // 引入Pinia的持久化插件
import Prism from "prismjs"; // 引入Prism代码高亮工具
import "swiper/css"; // 引入Swiper样式

import "swiper/css/autoplay"; // 引入Swiper自动播放样式
  import "swiper/css/mousewheel"; // 引入Swiper鼠标滚轮样式
import "swiper/css/navigation"; // 引入Swiper导航按钮样式
import "swiper/css/pagination"; // 引入Swiper分页器样式
import VueViewer from "v-viewer"; // 引入VueViewer图片查看器
import "viewerjs/dist/viewer.css"; // 引入VueViewer样式
import "virtual:svg-icons-register"; // 引入SVG图标

import { createApp, Directive } from "vue"; // 引入Vue相关依赖
import lazyPlugin from "vue3-lazy"; // 引入Vue3懒加载插件
import App from "./App.vue"; // 引入根组件App.vue
import error from "./assets/images/404.gif"; // 引入404错误页面图片
import loading from "./assets/images/loading.gif"; // 引入加载中图片

const app = createApp(App); // 创建Vue应用实例

//引入Element  plus以及原生组件样式文件
import ElementPlus from 'element-plus';
import 'element-plus/theme-chalk/index.css';
import locale from 'element-plus/lib/locale/lang/zh-cn'
app.use(ElementPlus, { locale })

Object.keys(directive).forEach((key) => {
  app.directive(key, (directive as { [key: string]: Directive })[key]); // 注册自定义指令
});
const pinia = createPinia(); // 创建Pinia实例
pinia.use(piniaPluginPersistedstate); // 使用Pinia的持久化插件
VMdPreview.use(vuepressTheme, {
  Prism,
})
  .use(createTodoListPlugin()) // 使用v-md-editor的todo-list插件
  .use(createKatexPlugin()); // 使用v-md-editor的katex插件
app.use(VMdPreview); // 使用v-md-editor的预览模块
app.use(naive); // 使用Naive UI组件库
app.use(pinia); // 使用Pinia状态管理库
app.use(router); // 使用路由配置
app.use(VueViewer); // 使用VueViewer图片查看器
app.use(lazyPlugin, {
  loading,
  error,
});



app.component("svg-icon", SvgIcon); // 注册SvgIcon组件
app.mount("#app"); // 挂载应用实例到DOM节点
titleChange(); // 调用标题变更函数