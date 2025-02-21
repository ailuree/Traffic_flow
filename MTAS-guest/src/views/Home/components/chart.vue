<!--
Bootstrap 的五个预设响应尺寸分别是：

xs - Extra Small（特小屏幕）：适用于小型移动设备，如手机。一般小于576px宽度。
sm - Small（小屏幕）：适用于较小的设备，如平板电脑或较小的桌面显示器。一般在576px到768px之间。
md - Medium（中等屏幕）：适用于中等尺寸的设备，如普通的桌面显示器或笔记本电脑。一般在768px到992px之间。
lg - Large（大屏幕）：适用于较大的设备，如较大的桌面显示器。一般在992px到1200px之间。
xl - Extra Large（特大屏幕）：适用于非常大尺寸的设备，如大型显示器或电视。一般大于1200px宽度。

-->
<template>
    <div class="my-chart">
        <el-row :gutter="24">
            <!-- 第一行 -->
            <!-- <el-col :xs="24" :sm="24" :md="24" :lg="12" :xl="12">
                <div class="chartCard " />
            </el-col>
            <el-col :xs="24" :sm="24" :md="24" :lg="12" :xl="12">
                <div class="chartCard " />
            </el-col> -->

            <!-- 第二行 -->
            <el-col :xs="24" :sm="24" :md="24" :lg="8" :xl="8">
                <div class="chartCard " style='width:100%; height:20rem; padding: 0.3rem;'>
                    <div id='lineChart' style='width:100%; height:98%; padding-top: 1rem;'></div>
                </div>
            </el-col>
            <el-col :xs="24" :sm="24" :md="24" :lg="8" :xl="8">
                <div class="chartCard " style='width:100%; height:20rem; padding: 0.3rem;'>
                    <div id='barChart' style='width:100%; height:98%; padding-top: 1rem;'></div>
                </div>
            </el-col>
            <el-col :xs="24" :sm="24" :md="24" :lg="8" :xl="8">
                <div class="chartCard " style='width:100%; height:20rem; padding: 0.3rem;'>
                    <div id='pieChart' style='width:100%; height:98%; padding-top: 1rem;'></div>
                </div>
            </el-col>
        </el-row>
    </div>
</template>

// 假设的API URL（你需要替换为实际的API地址）
const apiUrl = 'https://api.example.com/traffic-data';

<script setup lang="ts">
import { ref, onMounted, toRef } from "vue";
import * as echarts from "echarts";

onMounted(() => {
    type EChartsOption = echarts.EChartsOption;

 // 折线图
 var chartDomLineChart = document.getElementById('lineChart')!;
    var lineChart = echarts.init(chartDomLineChart);
    var option: EChartsOption;
    option = {
        color: ['#80FFA5'],
        title: {
            text: '🚀2024年国庆连霍高速车流量折线图',
            // left: 'center'
            // subtext: '每分钟数据'
        },

        tooltip: {

            trigger: 'axis',
            axisPointer: {
                type: 'cross',
                label: {
                    backgroundColor: '#80FFA5'
                }
            },
            formatter: '{b}:<br/> 车流量：{c} %'
        },
        legend: {
            data: ['车流量'],
            bottom: 0
        },
        // 保存
        toolbox: {

            feature: {
                saveAsImage: {
                    pixelRatio: 2, // 设置保存图像的像素比例，默认为1，可以提高清晰度
                    title: '下载', // 保存图像按钮的鼠标悬停标题
                    // icon: 'image://path/to/save-icon.png', // 自定义保存图像按钮的图标
                    name: '2024年国庆连霍高速车流量折线图', // 指定保存图像时使用的文件名
                    // backgroundColor: 'transparent', // 保存的图像背景颜色，默认为透明
                    excludeComponents: ['toolbox'], // 排除不想保存的组件，默认不排除任何组件
                    show: true, // 是否显示保存图像按钮，默认为true
                    // emphasis: {
                    //     show: true, // 鼠标悬停按钮时是否高亮显示，默认为true
                    //     iconStyle: {
                    //         textPosition: 'bottom',
                    //         color: '#000',
                    //         borderColor: '#000',
                    //         borderWidth: 1
                    //     }
                    // }
                }
            }
        },


        grid: {
            left: '5%',
            right: '5%',
            top: '20%',
            bottom: '10%',
            containLabel: true
        },

        xAxis: [{
            boundaryGap: false,
            type: 'category',
            data: ['10.1', '10.2', '10.3', '10.4', '10.5', '10.6', '10.7']
        }],
        yAxis: [{
            type: 'value'
        }],
        series: [
            {
                name: '车流量',
                type: 'line',
                // smooth: true,   // 是否为曲线
                // symbol: 'circle',
                // symbolSize: 5,
                //showSymbol: false,
                data: [92, 83, 71, 78, 65, 81, 74],
                areaStyle: {
                    opacity: 0.8,
                    color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                        {
                            offset: 0,
                            color: 'rgb(128, 255, 165)'
                        },
                        {
                            offset: 1,
                            color: 'rgb(1, 191, 236)'
                        }
                    ])
                },
                emphasis: {
                    focus: 'series'
                },

            },
            {
                polyline: true,
                // showSymbol: false,
                name: "流动光线",
                type: "lines",
                coordinateSystem: "cartesian2d",
                effect: {
                    delay: 100, // 延迟100ms开始流动
                    trailLength: 0.5,
                    show: true,
                    period: 5,
                    symbolSize: 4,
                    loop: true,
                },
                lineStyle: {
                    color: "#20db9df0",
                    width: 0,
                    opacity: 0,
                    curveness: 0.3, // 设置曲率
                    // type: "curve", // 设置为曲线

                },
                data: [
                    {
                        coords: [[0, 92], [1, 83], [2, 71], [3, 78], [4, 65], [5, 81], [6, 74]]
                    },
                ]
            },

        ]
    };
    option && lineChart.setOption(option);
    // 监听页面宽度改变事件
    // window.addEventListener("resize", handleResize);
    window.addEventListener("resize", function () {
        lineChart.resize();
    });


    // 柱状图
    var chartDomBarChart = document.getElementById('barChart')!;
    var barChart = echarts.init(chartDomBarChart);

    // 柱状图数据
    // prettier-ignore
    let dataAxis = ['10.1', '10.2', '10.3', '10.4', '10.5', '10.6', '10.7', ];
    // prettier-ignore

    let data = [72, 63, 51, 58, 45, 61, 54];
    let yMax = 100;
    let dataShadow = [];

    for (let i = 0; i < data.length; i++) {
        dataShadow.push(yMax);
    }

    option = {
        title: {
            text: '🚀2024年国庆连霍高速私家车车流量柱状图',
            // left: 'center'
            // subtext: 'Feature Sample: Gradient Color, Shadow, Click Zoom'
        },
        tooltip: {

            trigger: 'axis',
            axisPointer: {
                type: 'cross',
                label: {
                    backgroundColor: 'rgb(0, 221, 255)'
                }
            },
            formatter: '{b}:<br/> 车流量：{c} %'
        },
        legend: {
            data: ['车流量'],
            bottom: 0
        },
        // 保存
        toolbox: {

            feature: {
                saveAsImage: {
                    pixelRatio: 2, // 设置保存图像的像素比例，默认为1，可以提高清晰度
                    title: '下载', // 保存图像按钮的鼠标悬停标题
                    // icon: 'image://path/to/save-icon.png', // 自定义保存图像按钮的图标
                    name: ' 2024年国庆连霍高速私家车车流量柱状图', // 指定保存图像时使用的文件名
                    // backgroundColor: 'transparent', // 保存的图像背景颜色，默认为透明
                    excludeComponents: ['toolbox'], // 排除不想保存的组件，默认不排除任何组件
                    show: true, // 是否显示保存图像按钮，默认为true
                    // emphasis: {
                    //     show: true, // 鼠标悬停按钮时是否高亮显示，默认为true
                    //     iconStyle: {
                    //         textPosition: 'bottom',
                    //         color: '#000',
                    //         borderColor: '#000',
                    //         borderWidth: 1
                    //     }
                    // }
                }
            }
        },

        xAxis: {
            data: dataAxis,
            axisLabel: {
                color: '#999'
            }

        },
        yAxis: {
            axisLine: {
                show: false
            },
            axisTick: {
                show: false
            },
            axisLabel: {
                color: '#999'
            }
        },
        grid: {
            left: '5%',
            right: '5%',
            top: '20%',
            bottom: '10%',
            containLabel: true
        },
        dataZoom: [
            {
                type: 'inside'
            }
        ],
        series: [
            {
                name: '车流量',
                type: 'bar',
                showBackground: true,
                itemStyle: {
                    color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                        { offset: 0, color: 'rgb(0, 221, 255)' },
                        // { offset: 0.5, color: 'rgb(0, 221, 255)' },
                        { offset: 1, color: 'rgb(77, 119, 255)' }
                        // { offset: 0, color: 'rgb(6,190,179)' },
                        // // { offset: 0.5, color: 'rgb(0, 221, 255)' },
                        // { offset: 1, color: 'rgb(1,141,160)' }
                    ])
                },
                emphasis: {
                    itemStyle: {
                        color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                            { offset: 0, color: '#2378f7' },
                            { offset: 0.7, color: '#2378f7' },
                            { offset: 1, color: '#83bff6' }
                        ])
                    }
                },
                data: data
            }
        ]
    };

    // Enable data zoom when user click bar.
    const zoomSize = 6;
    barChart.on('click', function (params) {
        console.log(dataAxis[Math.max(params.dataIndex - zoomSize / 2, 0)]);
        barChart.dispatchAction({
            type: 'dataZoom',
            startValue: dataAxis[Math.max(params.dataIndex - zoomSize / 2, 0)],
            endValue:
                dataAxis[Math.min(params.dataIndex + zoomSize / 2, data.length - 1)]
        });
    });
    option && barChart.setOption(option);
    // 监听页面宽度改变事件
    window.addEventListener("resize", function () {
        barChart.resize();
    });

    // 饼状图
    var chartDomPieChart = document.getElementById('pieChart')!;
    var pieChart = echarts.init(chartDomPieChart);

    option = {
        title: {
            text: '🚀2024年国庆连霍高速车辆类型统计图',
            // subtext: 'Fake Data',
            // left: 'center'
        },
        tooltip: {
            trigger: 'item',
            formatter: '{b}:<br/> 车流量：{c} %'
        },
        legend: {
            bottom: 10,
            left: 'center',
            data: ['私家车', '面包车', '大客车', '大卡车']
        },
        series: [
            {
                type: 'pie',
                radius: '65%',
                center: ['50%', '50%'],
                selectedMode: 'single',
                data: [
                    {
                        value: 13, name: '大卡车', itemStyle: {
                            color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                                { offset: 0, color: 'rgb(89,253,193)' },
                                { offset: 1, color: 'rgb(48,214,231)' }
                            ])

                        },
                    },
                    {
                        value: 47, name: '私家车',
                        itemStyle: {
                            color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                                { offset: 0, color: 'rgb(37,210,234)' },
                                { offset: 1, color: 'rgb(8,209,255)' }
                            ])
                        },
                    },
                    {
                        value: 25, name: '面包车',
                        itemStyle: {
                            color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                                { offset: 0, color: 'rgb(148,107,244)' },
                                { offset: 1, color: 'rgb(151,162,247)' }
                            ])
                        },
                    },
                    {
                        value: 15, name: '大客车',
                        itemStyle: {
                            color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                                { offset: 0, color: 'rgb(61,185,249)' },
                                { offset: 1, color: 'rgb(98,106,241)' }
                            ])
                        },
                    },

                ],

                emphasis: {
                    itemStyle: {
                        shadowBlur: 10,
                        shadowOffsetX: 0,
                        shadowColor: 'rgba(0, 0, 0, 0.02)'
                    }
                }
            }
        ]
    };
    option && pieChart.setOption(option);
    // 监听页面宽度改变事件
    window.addEventListener("resize", function () {
        pieChart.resize();
    });
    // 雷达图


});
// function handleResize() {
//     var chartDom = document.getElementById('showorders')!;
//     var myChart = echarts.init(chartDom);
//     myChart.resize();
// }




</script>



<style>
.my-chart {
    width: 80%;
    margin: auto;
    margin-top: 10vh;
    margin-bottom: 10vh;
}

.el-row {
    margin-bottom: 20px;
}

.el-row:last-child {
    margin-bottom: 0;
}

.el-col {
    margin-top: 20px;
    border-radius: 4px;
}

.chartCard {
    border-radius: 4px;
    min-height: 36px;
    border-radius: 15px;
    /* box-shadow: 0 3px 3px 3px rgba(0, 0, 0, 0.04); */
    transition: all 0.2s ease-in-out;
    border: 1px solid rgba(181, 233, 248, 0.878);
}

.chartCard:hover {
    transform: translateY(-0.5px);
    box-shadow: 0 0 20px rgba(23, 196, 185, 0.2);
    border: 1px solid rgba(40, 189, 230, 0.878);
}
</style>
