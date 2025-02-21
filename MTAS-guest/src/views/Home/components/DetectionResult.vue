<template>
    <div class="detection-result">
        <div class="result-image">
            <img :src="props.afterImgPath" alt="检测结果" />
            <!-- 在图片上叠加标注框 -->
            <div v-for="label in props.labels" :key="label.id" class="bounding-box" :style="{
                left: `${label.x1}px`,
                top: `${label.y1}px`,
                width: `${label.x2 - label.x1}px`,
                height: `${label.y2 - label.y1}px`
            }">
                <span class="label-text">
                    ID: {{ label.id }} | {{ label.class }} | 置信度: {{ label.cf }}
                </span>
            </div>
        </div>

        <div class="detection-info">
            <h3>检测结果</h3>
            <p>总检测目标数：{{ props.totalObjects }}</p>
            <div class="labels-list">
                <div v-for="label in props.labels" :key="label.id" class="label-item">
                    <p>ID: {{ label.id }}</p>
                    <p>类别: {{ label.class }}</p>
                    <p>置信度: {{ label.cf }}</p>
                    <p>位置: ({{ label.x1 }}, {{ label.y1 }}) - ({{ label.x2 }}, {{ label.y2 }})</p>
                </div>
            </div>
        </div>
    </div>
</template>

<script setup lang="ts">
interface Label {
    id: string;
    class: string;
    cf: string;
    x1: string;
    y1: string;
    x2: string;
    y2: string;
}

interface Props {
    labels: Label[];
    afterImgPath: string;
    totalObjects: number;
}

const props = defineProps<Props>();
</script>

<style scoped lang="scss">
.detection-result {
    display: flex;
    gap: 20px;
    padding: 20px;

    .result-image {
        position: relative;

        img {
            max-width: 100%;
            height: auto;
        }

        .bounding-box {
            position: absolute;
            border: 2px solid #00ff00;

            .label-text {
                position: absolute;
                top: -25px;
                left: 0;
                background: rgba(0, 255, 0, 0.7);
                padding: 2px 5px;
                font-size: 12px;
                color: white;
            }
        }
    }

    .detection-info {
        flex: 1;

        .labels-list {
            max-height: 400px;
            overflow-y: auto;

            .label-item {
                background: #f5f5f5;
                padding: 10px;
                margin-bottom: 10px;
                border-radius: 4px;
            }
        }
    }
}
</style>