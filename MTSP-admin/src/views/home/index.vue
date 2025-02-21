<template>
  <div class="dashboard-container">
    <el-row :gutter="40" class="panel-group">
      <el-col :xs="12" :sm="12" :lg="6" class="card-panel-col">
        <div class="card-panel">
          <div class="card-panel-icon-wrapper icon-view">
            <svg-icon icon-class="dashboard" size="4em" class-tagName="card-panel-icon" />
          </div>
          <div class="card-panel-description">
            <div class="card-panel-text">
              车流统计总量
            </div>
            <span class="card-panel-num">5.20万辆</span>
          </div>
        </div>
      </el-col>
      <el-col :xs="12" :sm="12" :lg="6" class="card-panel-col">
        <div class="card-panel">
          <div class="card-panel-icon-wrapper icon-money">
            <svg-icon icon-class="date" size="3rem" class-tagName="card-panel-icon" />
          </div>
          <div class="card-panel-description">
            <div class="card-panel-text">
              系统运行时间
            </div>
            <span class="card-panel-num">888h</span>
          </div>
        </div>
      </el-col>

      <el-col :xs="12" :sm="12" :lg="6" class="card-panel-col">
        <div class="card-panel">
          <div class="card-panel-icon-wrapper icon-message">
            <svg-icon icon-class="button" size="4em" class-tagName="card-panel-icon" />
          </div>
          <div class="card-panel-description">
            <div class="card-panel-text">
              调用接口次数
            </div>
            <span class="card-panel-num">1.23K次</span>
          </div>
        </div>
      </el-col>

      <el-col :xs="12" :sm="12" :lg="6" class="card-panel-col">
        <div class="card-panel">
          <div class="card-panel-icon-wrapper icon-people">
            <svg-icon icon-class="peoples" size="4em" lass-tagName="card-panel-icon" />
          </div>
          <div class="card-panel-description">
            <div class="card-panel-text">
              访客总量
            </div>
            <span class="card-panel-num">1314人次</span>
          </div>
        </div>
      </el-col>

    </el-row>

    <el-row :gutter="32">
      <el-col :xs="24" :sm="24" :lg="8">
        <div class="chart-wrapper">
          <div class="title">上周车流量统计🚀</div>
          <Echarts :options="carFlow" height="350px"></Echarts>
        </div>
      </el-col>
      <el-col :xs="24" :sm="24" :lg="8">
        <div class="chart-wrapper">
          <div class="title">上周车辆分类统计🍉</div>
          <Echarts :options="category" height="350px"></Echarts>
        </div>
      </el-col>
      <el-col :xs="24" :sm="24" :lg="8">
        <div class="chart-wrapper">
          <div class="title">上周接口调用统计🌈</div>
          <Echarts :options="interfaceView" height="350px"></Echarts>
        </div>
      </el-col>
    </el-row>
  </div>
</template>

<script setup lang="ts">
import { TagVO } from "@/api/article/types";
import TagCloud from "@/components/TagCloud/index.vue";
import { onMounted, reactive, ref } from "vue";
const tagList = ref<TagVO[]>([]);
const viewCount = ref(500);
const messageCount = ref(0);
const userCount = ref(0);
const tagLoad = ref(false);
const articleCount = ref(0);


let interfaceView = reactive({
  xAxis: {
    data: ["1", "2", "3", "4", "5", "6", "7"] as string[],
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

  series: [
    {
      name: '接口调用',
      itemStyle: {
        color: 'rgb(64,201,198)'
      },
      lineStyle: {
        color: 'rgb(64,201,198)',
        width: 2
      },
      smooth: true,
      type: 'line',
      data: [
         18, 20, 27, 30, 33, 35, 39 // 这里是根据实际情况填充的数据
      ] as number[],
      animationDuration: 2800,
      animationEasing: 'cubicInOut'
    },
  ]
});
let category = reactive({
  tooltip: {
    trigger: 'item',
    formatter: '{a} <br/>{b} : {c} ({d}%)'
  },
  legend: {
    top: "bottom",
  },
  series: [
    {
      name: '分类统计',
      type: 'pie',
      radius: [15, 95],
      center: ['50%', '38%'],
      roseType: 'area',
      itemStyle: {
        borderRadius: 6
      },
      data: [{ value: 114, name: '小型车' },
      { value: 134, name: '中型车' },
      { value: 124, name: '大型车' },
      ] as {
        value: number;
        name: string;
      }[],
    }
  ]
});
let carFlow = reactive({
  tooltip: {
    trigger: 'axis',
    axisPointer: {
      type: 'shadow'
    }
  },
  color: ['#58AFFF'],
  grid: {
    left: '0%',
    right: '0%',
    bottom: '0%',
    top: '10%',
    containLabel: true
  },
  xAxis: {
    data: ["1", "2", "3", "4", "5", "6", "7",] as string[],
    axisTick: {
      alignWithLabel: true
    }
  },
  yAxis: {
    type: 'value',
    axisTick: {
      show: false
    }
  },
  series: [
    {
      name: '数据',
      type: 'bar',
      data: [113, 126, 139, 116, 104, 132, 147] as number[]
    }
  ]
});
const getList = () => {

};
onMounted(() => {
  getList();
})
</script>
<style lang="scss" scoped>
.title {
  font-size: 14px;
  color: var(--el-text-color-secondary);
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
