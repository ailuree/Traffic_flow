<template>
  <!-- 预测部分 -->
  <div class="content">
    <div class="card card-float">
      <!-- 模式选择 -->
      <div class="card-title">
        模式选择
      </div>
      <div class="card-child">
        <div style="margin-top: 2px">
          <el-radio-group v-model="radio1">
            <el-radio label="1" border class="radio">上传图片</el-radio>
            <el-radio label="2" border class="radio">摄像模式</el-radio>
            <!-- <el-radio label="2" border>打开摄像头</el-radio> -->
          </el-radio-group> 
        </div>
      </div>
      <hr class="divider">

      <!-- 图片上传 -->
      <div v-if="radio1 === '1'">
        <!-- <div class="card-title" style="margin-top: 10px;">
          图片上传
        </div> -->
        <div class="card-child">
          <Myupload ref="upload" @clickChild="clickEven"></Myupload>
        </div>
      </div>


      <!-- 摄像头功能 -->
      <div v-if="radio1 === '2'">
        <!-- <div class="card-title" style="margin-top: 10px;">
          开启摄像
        </div> -->
        <div class="card-child">
          <Mycamera @message="handleMessage"></Mycamera>
        </div>
      </div>

    </div>



    <!-- 预测结果 -->
    <div class="card card-float">
      <div class="card-title">
        预测情况
      </div>
      <div style="padding: 10px; width: 100%; height: 90%;">
        <img :src="after_img_path" class="pre-img" ref="imageRef">
      </div>
    </div>
  </div>

  <!-- 标签部分 -->
  <div class="content-table" name="bounce">
    <div>
      <el-table :data="mytableData" style="width: 100%" :row-class-name="tableRowClassName">
        <el-table-column prop="id" label="标签ID" width="180" />
        <el-table-column prop="class" label="类别" width="180" />
        <el-table-column prop="cf" label="置信度" width="180" />
        <el-table-column prop="x1" label="坐标(x1)" width="180" />
        <el-table-column prop="y1" label="坐标(y1)" width="180" />
        <el-table-column prop="x2" label="坐标(x2)" width="180" />
        <el-table-column prop="y2" label="坐标(y2)" width="180" />
      </el-table>
    </div>
  </div>

  <!-- 数据部分 -->
  <Mychart></Mychart>


  <!-- 侧边栏 -->
  <div class="config-box" v-show="sidebarVisible">

    <!-- 配置部分 -->
    <div class="config-tool">
      <!-- 车流阈值 -->
      <div class="config_title" style="margin-top: 8px;">
        车流阈值
      </div>
      <div class="card-child">
        <el-slider v-model="Slidervalue1" show-input size="small" />
      </div>

      <!-- 交互比 -->
      <div class="config_title" style="margin-top: 8px;">
        交互比
      </div>
      <div class="card-child">
        <el-slider v-model="Slidervalue2" show-input size="small" />
      </div>

      <!-- 置信度 -->
      <div class="config_title" style="margin-top: 8px;">
        置信度
      </div>
      <div class="card-child">
        <el-slider v-model="Slidervalue3" show-input size="small" />
      </div>



    </div>

    <!-- 侧边栏右边箭头 -->
    <div class="right-arrow">
      <div class="fuhao">
        &gt;
      </div>
    </div>
  </div>


</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue';
import Myupload from "../el-components/upload.vue";
import Mychart from "../components/chart.vue";
import Mycamera from "../components/camera.vue";
import DetectionResult from '../components/DetectionResult.vue';

import afterImgPath from "@/assets/images/predit111.jpg";   // 图片引入

import { ElMessage } from "element-plus";

import { detectImage, detectCamera } from '@/api/detection';
import type { DetectionResult as DetectionResultType, Label } from '@/api/detection/types';

// import Mylive from "../el-components/live2d.vue"

const radio1 = ref('1')

// ---------------------------------------------------------------滑块 config
const Slidervalue1 = ref(0)
const Slidervalue2 = ref(0)
const Slidervalue3 = ref(0)

// ---------------------------------------------------------------警告框TS
interface Alert {
  title: string;
  type: string;
}

const alerts = ref<Alert[]>([]);

function showAlerts(msg: string, type: string) {

  if (alerts.value.length > 0) {
    // 如果队列中已经有警报，则移除第一个警报
    alerts.value.shift();
  }

  const newAlert: Alert = {
    title: msg,
    type: type,
  };
  // 将新的 Alert 对象添加到数组末尾
  alerts.value.push(newAlert);

  setTimeout(() => {
    alerts.value = [];
  }, 2000); // 持续显示 3 秒后隐藏
}

// ---------------------------------------------------------------照相机TS
// 声明一个 ref 用于接收子组件传递的信息
const message = ref('');

// 自定义事件处理函数，用于接收子组件传递的信息
const handleMessage = async (photo: string) => {
  try {
    const { data } = await detectCamera(photo);
    if (data.code === 200) {
      detectionResult.value = data.data;
      // 更新图片路径
      after_img_path = data.data.after_img_path;
      // 更新表格数据
      mytableData.value = data.data.labels;
    }
  } catch (error) {
    console.error('识别失败:', error);
    ElMessage({
      message: '识别失败，请稍后重试',
      type: 'error'
    });
  }
};

// ---------------------------------------------------挂载 监听器


const sidebarVisible = ref(false);

onMounted(() => {
  // 侧边栏 点击出现
  const music = document.querySelector<HTMLElement>(".config-box");
  const arrow = document.querySelector<HTMLElement>(".right-arrow");
  const fuhao = document.querySelector<HTMLElement>(".fuhao");
  let isFlipped = false;

  const arrowClickHandler = () => {
    if (music && arrow && fuhao) {
      if (isFlipped) {
        music.classList.remove("show");
        fuhao.classList.remove("flip");
        isFlipped = false;
        arrow.setAttribute("transform", "rotate(0)");
      } else {
        isFlipped = true;
        music.classList.add("show");
        fuhao.classList.add("flip");
        arrow.setAttribute("transform", "rotate(180)");
      }
    }
  };

  arrow?.addEventListener("click", arrowClickHandler);

  // 出现监听器
  const updateSidebarVisibility = () => {
    sidebarVisible.value = window.pageYOffset > 500;
    // console.log(sidebarVisible.value)
  };
  window.addEventListener('scroll', updateSidebarVisibility);


});



const rownumber = ref(0)



const tableRowClassName = ({
  row,
  rowIndex,
}: {
  row: Label
  rowIndex: number
}) => {
  if (rownumber.value === 0) {
    rownumber.value = 1
    return 'primary-row'
  } else if (rownumber.value === 1) {
    rownumber.value = 2
    return ''
  } else if (rownumber.value === 2) {
    rownumber.value = 3
    return 'success-row'
  }
  rownumber.value = 0
  return ''
}


let mytableData = ref<Label[]>([

  // {
  //   id: '1',
  //   class: 'car',
  //   cf: '0.5',
  //   x1: '200',
  //   y1: '100',
  //   x2: '200',
  //   y2: '100',
  // },
]);


let after_img_path: string = afterImgPath
let labels: any[];


// ---------------------------------------------------------------后端返回数据处理
const clickEven = (val: { code: number, data: any, msg: string }) => {
  try {
    if (val.data === '照片中没有目标物体哟！') {
      ElMessage({
        message: '照片中没有目标物体哟！',
        type: "warning",
        duration: 5 * 1000,
      });
      return;
    }
    
    if (val.msg === '服务器繁忙，请稍后再试！') {
      ElMessage({
        message: '服务器繁忙，请稍后再试！',
        type: "error",
        duration: 5 * 1000,
      });
      return;
    }
    
    if (val.msg === '执行成功！') {
      ElMessage({
        message: '识别成功！',
        type: "success",
        duration: 5 * 1000,
      });
    }

    // 更新图片
    if (imageRef.value && val.data.after_img_path) {
      imageRef.value.src = val.data.after_img_path;
      after_img_path = val.data.after_img_path;
    }

    // 处理标签数据
    if (val.data.labels && Array.isArray(val.data.labels)) {
      const rowData: Label[] = val.data.labels.map((label: string) => {
        // 假设后端返回的格式是: "ID: 1 Class: car Confidence: 0.95 Coordinates: 100 200 300 400"
        const parts = label.split(' ').filter(Boolean);
        return {
          id: parts[1],                    // ID值
          class: parts[3],                 // 类别
          cf: parts[5],                    // 置信度
          x1: parts[7],                    // x1坐标
          y1: parts[8],                    // y1坐标
          x2: parts[9],                    // x2坐标
          y2: parts[10].replace(/[,)]$/, '') // y2坐标，移除可能的逗号或括号
        };
      });

      // 更新表格数据
      mytableData.value = rowData;
      console.log('处理后的数据:', mytableData.value);
    } else {
      console.warn('未收到有效的标签数据');
    }

  } catch (error) {
    console.error('数据处理错误:', error);
    ElMessage({
      message: '数据处理出错，请重试',
      type: 'error',
      duration: 5 * 1000
    });
  }
};



// ---------------------------------------------------------------图片自适应
const imageRef = ref<HTMLImageElement | null>(null);
const adjustImageSize = () => {
  if (imageRef.value && imageRef.value.naturalHeight > imageRef.value.naturalWidth) {
    imageRef.value.style.height = '100%';

  }
};

adjustImageSize();

const detectionResult = ref(null);

// 修改handleImageUpload函数
const handleImageUpload = async (file: File) => {
  try {
    const { data } = await detectImage(file);
    if (data.code === 200) {
      detectionResult.value = data.data;
      // 更新图片路径
      after_img_path = data.data.after_img_path;
      // 更新表格数据
      mytableData.value = data.data.labels;
    }
  } catch (error) {
    console.error('识别失败:', error);
    ElMessage({
      message: '识别失败，请稍后重试',
      type: 'error'
    });
  }
};

</script>


<style scoped>
.content {
  display: flex;
  flex-direction: column;
  width: 80%;
  /* height: 80vh; */
  margin: auto;
  margin-top: 10vh;
  margin-bottom: 10vh;
  user-select: none;
}

@media (min-width: 600px) {
  .content {
    flex-direction: row;
  }
}

.content-table {
  display: flex;
  flex-direction: column;
  width: 80%;
  /* height: 80vh; */
  margin: auto;
  margin-top: 10vh;
  margin-bottom: 10vh;
}

.card {
  flex: 1;
  padding: 1rem;
  margin: 1rem;
  border-radius: 15px !important;
  /* box-shadow: 4px 4px 2px rgba(131, 220, 238, 0.284); */
  transition: all 0.3s ease-in-out;
  border: 1px solid rgba(181, 233, 248, 0.878);

}

.card-title {
  font-size: 1.3vw;
  font-weight: bold;
}

.card-child {
  padding: 0.5rem;
}

/* 卡片悬停 */
.card-float:hover {
  transform: translateY(-1px);
  box-shadow: 0 0 20px rgba(23, 196, 185, 0.2);
  border: 1px solid rgba(40, 189, 230, 0.878);
}

.config_title {
  font-size: 1vw;
  font-weight: bold;
}


.pre-img {
  height: auto;
  width: 100%;
}
</style>

<!-- 表格CSS -->
<style>
.el-table .primary-row {
  --el-table-tr-bg-color: var(--el-color-primary-light-9);
}

.el-table .success-row {
  --el-table-tr-bg-color: var(--el-color-success-light-9);
}
</style>

<!-- 侧边栏CSS -->
<style>
/* 定义样式 */
.right-arrow {
  flex: 1;

  display: flex;
  justify-content: center;
  align-items: center;
  font-size: 30px;
  transition: all .2s ease-in-out;
  border-radius: 0px 1.25rem 1.25rem 0px;
  /* 无法选中 */
  user-select: none;
}

.right-arrow:hover {
  background-color: rgb(248, 253, 253);
  color: rgb(82, 196, 248);

}

/* 符号 > */
.fuhao {
  transition: all .3s ease-in-out;
  justify-content: center;
  align-items: center;
  font-size: 2rem;
  transition: all .2s ease-in-out;
}

.fuhao.flip {
  transform: rotate(-180deg);
}




.config-tool {
  flex: 8;
  position: relative;
  overflow: hidden;
  transition: all 0.5s ease-in-out;
  padding-left: 1rem;
  padding-right: 1rem;

}

.config-box {
  z-index: 9999;
  position: fixed;
  flex: 9;

  width: 400px;
  height: 250px;
  padding-left: 0.8rem;

  transform: scaleX(-1);
  transform: translateY(-50%);

  /* 左下角 */
  left: -360px;
  bottom: 18%;

  /* 添加阴影 */
  box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.2);
  background-color: rgb(248, 253, 253);
  /* 横向排列 */
  display: flex;
  flex-direction: row;

  border-radius: 0px 1.25rem 1.25rem 0px;

  /* 平滑 */
  transition: all .2s ease-in-out;
}

.config-box:hover {
  box-shadow: 0 0 20px rgba(23, 196, 185, 0.5);

}

.config-box.show {
  left: 0px;
  opacity: 1;
}


@media (max-width: 767px) {

  /* 手机设备 */
  .config-box {
    position: fixed;
    flex: 9;

    width: 300px;

    /* 左下角 */
    left: -270px;

    transform: scaleX(-1);
    transform: translateY(-50%);

    /* 添加阴影 */
    box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.2);
    /* 横向排列 */
    display: flex;
    flex-direction: row;
    border-radius: 10px;

    /* 平滑 */
    transition: all .2s ease-in-out;


  }

}


/* 分割线 */
.divider {
  margin: 0.5rem;
  border: none;
  height: 1px;
  background-color: rgb(161, 225, 241);
}
</style>


<!-- radio -->
<style>
@media (max-width: 600px) {
  .radio {
    margin: 0.5rem;
  }
}
</style>