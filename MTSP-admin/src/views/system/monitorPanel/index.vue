<template>
  <div class="dashboard-container">

    <el-row :gutter="32">
      <el-col :xs="24" :sm="24" :lg="16">
        <div class="chart-wrapper">
          <div class="title">视频检测 🚀</div>
          <!-- <Echarts :options="category" height="350px"></Echarts> -->
          <!-- 视频播放 -->
          <div style="height:470px; margin-top:20px">
            <video style="width:100%;height:100%; padding:10px; background-color: rgb(250, 253, 253);" muted autoplay
              id="video" ref="videoElement"></video>
          </div>
        </div>
      </el-col>
      <el-col :xs="24" :sm="24" :lg="8">
        <div class="chart-wrapper">
          <div class="title">配置表单 📋</div>
          <!-- 表单 -->
          <div style="height:470px; margin-top:20px">
            <el-form :model="form" label-width="80px">

              <el-form-item label="视频路径">
                <el-input v-model="route.query.video" :disabled="true" />
              </el-form-item>

              <el-form-item label="拉流路径">
                <el-input v-model="route.query.url" :disabled="true" />
              </el-form-item>

              <el-form-item label="配置人员">
                <el-select v-model="route.query.person" placeholder="请选择配置人员" :disabled="true">
                  <el-option label="Pan" value="Pan" />
                  <el-option label="Rock" value="Rock" />
                </el-select>
              </el-form-item>

              <el-form-item label="开启警报">
                <el-input v-model="route.query.is_alarm" :disabled="true" />
              </el-form-item>

              <el-form-item label="警报阈值">
                <el-input v-model="route.query.threshold" :disabled="true" />
              </el-form-item>

              <el-form-item label="监控地点">
                <el-input v-model="route.query.location" :disabled="true" />
              </el-form-item>

              <el-form-item label="模式选择">
                <el-radio-group v-model="route.query.mode" :disabled="true">
                  <el-radio label="快速模式" />
                  <el-radio label="精确模式" />
                </el-radio-group>
              </el-form-item>

              <el-form-item label="配置备注">
                <el-input v-model="route.query.remark" type="textarea" :disabled="true" />
              </el-form-item>

            </el-form>
          </div>
        </div>
      </el-col>
    </el-row>
    <el-row class="data-card">
      <div class="title">算法表现 ✨</div>
      <Echarts :options="algorithmView" height="350px"></Echarts>
    </el-row>
  </div>
</template>

<script setup lang="ts">
import { useRoute } from 'vue-router';
const route = useRoute();
import { ElMessage } from "element-plus";
import { onMounted, reactive, ref } from "vue";
import { submitMonitorForm } from '@/api/monitor';
import { wsPlayer } from '../streamingSetting/wsplayer/wsPlayer.js'; // 引入 wsPlayer.js 路径
const videoElement = ref(null);
let player = ref(null);
onMounted(() => {
  const player = videoElement.value;
  if (videoElement.value) {
    const player = new wsPlayer('video', form.videoUrl);
    player.open();
    console.log(11);
  }
});


// 表单
const form = reactive({
  videoUrl: 'ws://127.0.0.1/live/test.live.mp4',
  person: '',
  isAlarm: false,
  remark: '',
  mode: '',
  threshold: '',
  location: '',
})

// 提交表单的方法
const formSubmit = () => {
  console.log(111)
  // 调用 submitMonitorForm 方法提交表单数据
  submitMonitorForm(form)
    .then(response => {
      const result: Result<any> = response.data;
      // 处理返回的结果
      console.log(result);
      ElMessage({
        message: "提交成功",
        type: "success",
        duration: 2 * 1000,
        onClose: () => {
          window.location.reload();
        },
      });
    })
    .catch(error => {
      console.error(error);
      ElMessage({
        message: "提交失败，请稍后再试！",
        type: "danger",
        duration: 2 * 1000,
        onClose: () => {
          window.location.reload();
        },
      });
    });
};

let algorithmView = reactive({
  xAxis: {
    data: ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"] as string[],
    boundaryGap: false,
    axisTick: {
      show: false
    }
  },
  grid: {
    left: 8,
    right: 35,
    bottom: 0,
    top: 30,
    containLabel: true
  },
  tooltip: {
    trigger: 'axis',
    axisPointer: {
      type: 'cross'
    },
    padding: [5, 10]
  },
  yAxis: {
    axisTick: {
      show: false
    }
  },
  legend: {
    data: ['准确', '快速']
  },
  series: [
    {
      name: '准确',
      itemStyle: {
        color: '#FF005A'
      },
      lineStyle: {
        color: '#FF005A',
        width: 2
      },
      smooth: true,
      type: 'line',
      data: [
        10, 12, 13, 15, 17, 20, 22, 25, 28, 30, 32, 35 // 这里是根据实际情况填充的数据
      ] as number[],
      animationDuration: 2800,
      animationEasing: 'cubicInOut'
    },
    {
      name: '快速',
      smooth: true,
      type: 'line',
      itemStyle: {
        color: '#3888fa'
      },
      lineStyle: {
        color: '#3888fa',
        width: 2
      },
      areaStyle: {
        color: '#f3f8ff'
      },
      data: [
        5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16  // 这里是根据实际情况填充的数据
      ] as number[],
      animationDuration: 2800,
      animationEasing: 'quadraticOut'
    }
  ]
});


</script>
<style lang="scss" scoped>
.title {
  font-size: 18px;
  color: rgba(0, 0, 0, 0.836);
  font-weight: 700;
}

.data-card {
  background: var(--el-bg-color-overlay);
  padding: 1rem;
}

.dashboard-container {
  padding: 32px;
  background: var(--el-bg-color-page);
  position: relative;

  .github-corner {
    position: absolute;
    top: 0px;
    border: 0;
    right: 0;
  }

  .chart-wrapper {
    background: var(--el-bg-color-overlay);
    padding: 1rem;
    margin-bottom: 2rem;
  }
}

.panel-group {
  margin-top: 18px;

  .card-panel-col {
    margin-bottom: 32px;
  }

  .card-panel {
    height: 108px;
    cursor: pointer;
    font-size: 12px;
    position: relative;
    overflow: hidden;
    color: #666;
    background: var(--el-bg-color-overlay);
    box-shadow: 4px 4px 40px rgba(0, 0, 0, .05);
    border-color: rgba(0, 0, 0, .05);

    &:hover {
      .card-panel-icon-wrapper {
        color: #fff;
      }

      .icon-people {
        background: #40c9c6;
      }

      .icon-message {
        background: #36a3f7;
      }

      .icon-money {
        background: #f4516c;
      }

      .icon-view {
        background: #34bfa3
      }
    }

    .icon-people {
      color: #40c9c6;
    }

    .icon-message {
      color: #36a3f7;
    }

    .icon-money {
      color: #f4516c;
    }

    .icon-view {
      color: #34bfa3
    }

    .card-panel-icon-wrapper {
      float: left;
      margin: 14px 0 0 14px;
      padding: 16px;
      transition: all 0.38s ease-out;
      border-radius: 6px;
    }

    .card-panel-description {
      float: right;
      font-weight: bold;
      margin: 26px;
      margin-left: 0px;

      .card-panel-text {
        line-height: 18px;
        color: var(--el-text-color-secondary);
        font-size: 16px;
        margin-bottom: 12px;
      }

      .card-panel-num {
        font-size: 20px;
      }
    }
  }
}

@media (max-width:1024px) {
  .chart-wrapper {
    padding: 8px;
  }
}

@media (max-width:550px) {
  .card-panel-description {
    display: none;
  }

  .card-panel-icon-wrapper {
    float: none !important;
    width: 100%;
    height: 100%;
    margin: 0 !important;

    .svg-icon {
      display: block;
      margin: 14px auto !important;
      float: none !important;
    }
  }
}
</style>
