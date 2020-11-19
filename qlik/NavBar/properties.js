define([], function () {
	return {
		qKey: 'cubricks',
		type: 'items',
		component: 'accordion',
		items: {
			topMenu: {
				type: 'items',
				component: "expandable-items",
				label: 'Top Menu',
				items: {
					header1: {
						type: "items",
						label: "Logo and Title",
						items: {
							logoShow: {
								type: "boolean",
								component: "switch",
								label: "Show Logo",
								ref: "qProp.logoShow",
								defaultValue: true,
								options: [{
									value: true,
									label: "Show"
								}, {
									value: false,
									label: "Hide"
								}],
							},
							logoUrl: {
								label: 'Logo Url',
								ref: 'qProp.logoUrl',
								type: 'string',
								component: 'string',
							},
							logoImg: {
								label: "Logo",
								component: "media",
								layoutRef: "logoUrl",
								ref: "qProp.logoUrl",
								type: "string",
							},
							titleShow: {
								type: "boolean",
								component: "switch",
								label: "Show title",
								ref: "qProp.titleShow",
								defaultValue: false,
								options: [{
									value: true,
									label: "Show"
								}, {
									value: false,
									label: "Hide"
								}],
							}
						},
					}
				},
			},
			appearance: {
				uses: 'settings',
			},
		}
	};
});