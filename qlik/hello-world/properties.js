define([], function () {
    'use strict';
    // *****************************************************************************
    // Dimensions, Measures, Sorting
    // *****************************************************************************
    var dimensions = {
        uses: 'dimensions',
        min: 0,
        max: 1
    };
    var measures = {
        uses: 'measures',
        min: 0,
        max: 1
    };
    var sorting = {
        uses: 'sorting'
    };
    // *****************************************************************************
    // Custom section
    // *****************************************************************************
    var header1_item1 = {
        ref: "props.section1.item1",
        label: "Section 1 / Item 1",
        type: "string",
        expression: "optional"
    };
    var header1_item2 = {
        ref: "props.section1.item2",
        label: "Section 1 / Item 2",
        type: "string",
        expression: "optional"
    };
    var header2_item1 = {
        ref: "props.section2.item1",
        label: "Section 2 / Item 1",
        type: "string",
        expression: "optional"
    };
    var header2_item2 = {
        ref: "props.section2.item2",
        label: "Section 2 / Item 2",
        type: "number",
        expression: "optional"
    };

    var myCustomSection = {
        type: "items",
        component: "expandable-items",
        label: "My Accordion Section",
        items: {
            header1: {
                type: "items",
                label: "Header 1",
                items: {
                    header1_item1: header1_item1,
                    header1_item2: header1_item2
                }
            },
            header2: {
                type: "items",
                label: "Header 2",
                items: {
                    header2_item1: header2_item1,
                    header2_item2: header2_item2
                }
            }

        }
    }

    var myMedia = {
        label: "My media",
        component: "media",
        ref: "props.myMedia",
        layoutRef: "myMedia",
        type: "string"
    }

    // *****************************************************************************
    // Appearance section
    // *****************************************************************************
    var appearance = {
        uses: 'settings',
    };
    // *****************************************************************************
    // Main properties panel definition
    // Only what is defined here is returned from properties.js
    // *****************************************************************************
    return {
        type: 'items',
        component: 'accordion',
        items: {
            dimensions: dimensions,
            measures: measures,
            sorting: sorting,
            appearance: appearance,
            myCustomSection: myCustomSection,
            myMedia: myMedia,
            MySwitchProp: {
                type: "boolean",
                component: "switch",
                label: "Switch me On",
                ref: "props.scape",
                options: [{
                    value: true,
                    label: "On"
                }, {
                    value: false,
                    label: "Not On"
                }],
                defaultValue: true
            }
        }
    };
});