function ask_for_style_name(sd_model_checkpoint, dummy_component, _, resolution, val_and_checkpointing_steps, max_train_steps, steps_per_photos, train_batch_size, gradient_accumulation_steps, dataloader_num_workers, learning_rate, rank, network_alpha, validation, main_image, instance_images, enable_rl, max_rl_time, timestep_fraction) {
    var name_ = prompt('User id:');
    return [sd_model_checkpoint, dummy_component, name_, resolution, val_and_checkpointing_steps, max_train_steps, steps_per_photos, train_batch_size, gradient_accumulation_steps, dataloader_num_workers, learning_rate, rank, network_alpha, validation, main_image, instance_images, enable_rl, max_rl_time, timestep_fraction, ];
}
function samCreateDot(sam_image, image, coord, label) {
    const x = coord.x;
    const y = coord.y;
    const realCoord = samGetRealCoordinate(image, coord.x, coord.y);
    if (realCoord[0] >= 0 && realCoord[0] <= image.naturalWidth && realCoord[1] >= 0 && realCoord[1] <= image.naturalHeight) {
        const isPositive = label == (samTabPrefix() + "positive");
        const circle = document.createElement("div");
        circle.style.position = "absolute";
        circle.style.width = "10px";
        circle.style.height = "10px";
        circle.style.borderRadius = "50%";
        circle.style.left = x + "px";
        circle.style.top = y + "px";
        circle.className = label;
        circle.style.backgroundColor = isPositive ? "black" : "red";
        circle.title = (isPositive ? "positive" : "negative") + " point label, left click it to cancel.";
        sam_image.appendChild(circle);
        circle.addEventListener("click", e => {
            e.stopPropagation();
            circle.remove();
            if (gradioApp().querySelectorAll("." + samTabPrefix() + "positive").length != 0 ||
                gradioApp().querySelectorAll("." + samTabPrefix() + "negative").length != 0) {
                if (samIsRealTimePreview()) {
                    samImmediatelyGenerate();
                }
            }
        });
        if (samIsRealTimePreview()) {
            samImmediatelyGenerate();
        }
    }
}

function samRemoveDots() {
    const sam_image = gradioApp().getElementById(samTabPrefix() + "input_image");
    if (sam_image) {
        ["." + samTabPrefix() + "positive", "." + samTabPrefix() + "negative"].forEach(cls => {
            const dots = sam_image.querySelectorAll(cls);
    
            dots.forEach(dot => {
                dot.remove();
            });
        })
    }
    return arguments;
}

function create_submit_sam_args(args) {
    res = []
    for (var i = 0; i < args.length; i++) {
        res.push(args[i])
    }

    res[res.length - 1] = null

    return res
}


function submit_dino() {
    res = []
    for (var i = 0; i < arguments.length; i++) {
        res.push(arguments[i])
    }

    res[res.length - 2] = null
    res[res.length - 1] = null
    return res
}

function submit_sam() {
    let res = create_submit_sam_args(arguments);
    let positive_points = [];
    let negative_points = [];
    const sam_image = gradioApp().getElementById(samTabPrefix() + "input_image");
    const image = sam_image.querySelector('img');
    const classes = ["." + samTabPrefix() + "positive", "." + samTabPrefix() + "negative"];
    classes.forEach(cls => {
        const dots = sam_image.querySelectorAll(cls);
        dots.forEach(dot => {
            const width = parseFloat(dot.style["left"]);
            const height = parseFloat(dot.style["top"]);
            if (cls == "." + samTabPrefix() + "positive") {
                positive_points.push(samGetRealCoordinate(image, width, height));
            } else {
                negative_points.push(samGetRealCoordinate(image, width, height));
            }
        });
    });
    res[2] = positive_points;
    res[3] = negative_points;
    return res
}

samPrevImg = {
    "txt2img_sam_": null,
    "img2img_sam_": null,
}

onUiUpdate(() => {
    const sam_image = gradioApp().getElementById(samTabPrefix() + "input_image")
    if (sam_image) {
        const image = sam_image.querySelector('img')
        if (image && samPrevImg[samTabPrefix()] != image.src) {
            samRemoveDots();
            samPrevImg[samTabPrefix()] = image.src;

            image.addEventListener("click", event => {
                const rect = event.target.getBoundingClientRect();
                const x = event.clientX - rect.left;
                const y = event.clientY - rect.top;

                samCreateDot(sam_image, event.target, { x, y }, samTabPrefix() + "positive");
            });

            image.addEventListener("contextmenu", event => {
                event.preventDefault();
                const rect = event.target.getBoundingClientRect();
                const x = event.clientX - rect.left;
                const y = event.clientY - rect.top;

                samCreateDot(sam_image, event.target, { x, y }, samTabPrefix() + "negative");
            });

            const observer = new MutationObserver(mutations => {
                mutations.forEach(mutation => {
                    if (mutation.type === 'attributes' && mutation.attributeName === 'src' && mutation.target === image) {
                        samRemoveDots();
                        samPrevImg[samTabPrefix()] = image.src;
                    }
                });
            });

            observer.observe(image, { attributes: true });
        } else if (!image) {
            samRemoveDots();
            samPrevImg[samTabPrefix()] = null;
        }
    }
})